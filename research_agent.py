"""
AI Research Agent using LangGraph + OpenRouter
==============================================
A stateful, multi-step research agent that decomposes queries,
searches the web, extracts evidence, reflects on sufficiency,
and synthesizes cited answers.

Requirements:
    - OPENROUTER_API_KEY environment variable
    - Tavily API key (TAVILY_API_KEY) OR ddgs package for DuckDuckGo
"""

import os
import re
import time
import json
import operator
from typing import TypedDict, List, Optional, Literal, Annotated, Dict, Any
from datetime import datetime
from enum import Enum

import httpx
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# 1. CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

MAX_ITERATIONS = 3
MAX_SEARCH_RESULTS = 5
MAX_PAGES_PER_SUBQUESTION = 3
MAX_PAGE_TOKENS = 6000  # character budget per page


def create_llm(temperature: float = 0.1, max_tokens: int = 4000):
    """Create an LLM instance connected to OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY environment variable.")
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=OPENROUTER_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ============================================================
# 2. PYDANTIC MODELS (Structured Outputs)
# ============================================================

class SubQuestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    SKIPPED = "skipped"


class SubQuestion(BaseModel):
    """A single decomposed sub-question from the research plan."""
    id: str = Field(description="Unique identifier, e.g., 'sq1'")
    text: str = Field(description="The sub-question text")
    search_strategy: str = Field(
        description="Search strategy: 'general', 'news', 'academic', or 'domain'"
    )
    expected_evidence: str = Field(description="What type of evidence is expected")
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of sub-questions that must be answered first"
    )
    status: SubQuestionStatus = Field(default=SubQuestionStatus.PENDING)


class ResearchPlan(BaseModel):
    """Structured plan output from the planner node."""
    sub_questions: List[SubQuestion] = Field(
        description="Ordered list of sub-questions to research"
    )
    reasoning: str = Field(description="Why this decomposition was chosen")


class EvidenceItem(BaseModel):
    """A single piece of extracted evidence."""
    claim: str = Field(description="The factual claim or finding")
    source_url: str = Field(description="URL of the source")
    source_title: str = Field(default="", description="Title of the source page")
    snippet: str = Field(description="Exact quote or close paraphrase from source")
    relevance: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    credibility: float = Field(ge=0.0, le=1.0, description="Credibility score 0-1")
    sub_question_id: str = Field(description="Which sub-question this evidence addresses")
    retrieved_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ExtractedEvidence(BaseModel):
    """Structured output from the evidence extractor."""
    evidence_items: List[EvidenceItem] = Field(
        default_factory=list,
        description="Relevant evidence extracted from the page"
    )
    summary: str = Field(default="", description="Brief summary of what was found")


class ReflectionResult(BaseModel):
    """Structured output from the reflection node."""
    is_sufficient: bool = Field(description="Whether evidence is sufficient to answer the query")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in current evidence")
    gaps: List[str] = Field(
        default_factory=list,
        description="Specific gaps as new sub-question texts (empty if sufficient)"
    )
    reasoning: str = Field(description="Explanation of the assessment")
    new_sub_questions: List[SubQuestion] = Field(
        default_factory=list,
        description="New sub-questions to add to the plan (empty if sufficient)"
    )


class FinalAnswer(BaseModel):
    """Structured final answer with citations."""
    answer: str = Field(description="Comprehensive answer to the user's query")
    citations: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of {citation_id, url, title} objects"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence")
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up research questions"
    )
    caveats: str = Field(default="", description="Any caveats or limitations")


# ============================================================
# 3. STATE DEFINITION
# ============================================================

class ResearchState(TypedDict):
    """Shared state across all nodes in the research graph."""
    query: str
    plan: List[SubQuestion]
    current_sq_id: Optional[str]
    evidence: Annotated[List[EvidenceItem], operator.add]
    gaps: List[str]
    raw_search_results: List[Dict[str, Any]]
    fetched_pages: List[Dict[str, str]]
    iteration: int
    max_iterations: int
    status: Literal["planning", "searching", "reflecting", "done", "error"]
    error: Optional[str]
    final_answer: Optional[FinalAnswer]
    messages: Annotated[List[str], operator.add]  # for logging/observability


# ============================================================
# 4. SEARCH PROVIDERS
# ============================================================

class SearchProvider:
    """Base interface for search providers."""

    def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict[str, Any]]:
        raise NotImplementedError


class TavilySearchProvider(SearchProvider):
    """Search provider using Tavily API, optimized for news and recent content."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS, strategy: str = "general") -> List[Dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        
        # Use Tavily's News endpoint if the planner asked for news
        topic = "news" if strategy == "news" else "general"
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "topic": topic,
            "include_answer": False,
            # Optional: you can uncomment the line below to only get results from the last 7 days
            # "days": 7, 
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(self.endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                results = []
                for r in data.get("results", []):
                    results.append({
                        "url": r.get("url", ""),
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "source_type": "web",
                    })
                return results
        except Exception as e:
            print(f"  [Tavily] Search error: {e}")
            return []


class DuckDuckGoSearchProvider(SearchProvider):
    """Search provider using DuckDuckGo (no API key needed)."""

    # FIX: Added 'strategy: str = "general"' to accept the argument from search_node
    def search(self, query: str, max_results: int = MAX_SEARCH_RESULTS, strategy: str = "general") -> List[Dict[str, Any]]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                print("  [DuckDuckGo] Package not installed. Install with: pip install ddgs")
                return []

        try:
            results = []
            with DDGS() as ddgs:
                # Note: DuckDuckGo text search doesn't have a strict news filter like Tavily,
                # so we accept the strategy argument but use standard text search.
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "url": r.get("href", r.get("url", "")),
                        "title": r.get("title", ""),
                        "snippet": r.get("body", r.get("snippet", "")),
                        "source_type": "web",
                    })
            return results
        except Exception as e:
            print(f"  [DuckDuckGo] Search error: {e}")
            return []


def get_search_provider() -> SearchProvider:
    """Select the best available search provider."""
    if TAVILY_API_KEY:
        print("[Setup] Using Tavily search provider")
        return TavilySearchProvider(TAVILY_API_KEY)
    else:
        print("[Setup] Using DuckDuckGo search provider (no API key required)")
        return DuckDuckGoSearchProvider()


# ============================================================
# 5. PAGE FETCHER (Jina Reader)
# ============================================================

def fetch_page_content(url: str, max_chars: int = MAX_PAGE_TOKENS) -> str:
    """Fetch clean markdown content from a URL using Jina Reader."""
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Accept": "text/markdown",
        "X-Return-Format": "markdown",
    }
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(jina_url, headers=headers)
            if resp.status_code == 200:
                content = resp.text[:max_chars]
                return content
            else:
                print(f"  [Fetch] Jina returned {resp.status_code} for {url}")
                return ""
    except Exception as e:
        print(f"  [Fetch] Error fetching {url}: {e}")
        return ""


# ============================================================
# 6. LLM HELPER FUNCTIONS
# ============================================================

def call_llm_structured(
    llm: ChatOpenAI,
    system_prompt: str,
    user_prompt: str,
    output_model: type,
) -> Optional[BaseModel]:
    """Call LLM with structured output, with JSON fallback."""
    try:
        structured_llm = llm.with_structured_output(output_model)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        result = structured_llm.invoke(messages)
        return result
    except Exception as e:
        print(f"  [LLM] Structured output failed ({e}), trying JSON fallback...")
        try:
            schema = output_model.model_json_schema()
            json_instruction = (
                f"\n\nIMPORTANT: Respond with ONLY a valid JSON object matching this schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt + json_instruction),
            ]
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return output_model.model_validate_json(json_match.group())
        except Exception as e2:
            print(f"  [LLM] JSON fallback also failed: {e2}")
        return None


def call_llm_text(
    llm: ChatOpenAI,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call LLM and return raw text."""
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"  [LLM] Text call failed: {e}")
        return ""


# ============================================================
# 7. NODE IMPLEMENTATIONS
# ============================================================

def ingest_node(state: ResearchState) -> Dict[str, Any]:
    """Normalize input and apply basic guardrails."""
    print("\n{'='*60}")
    print("[Node: Ingest] Processing user query...")
    query = state["query"].strip()
    if not query:
        return {"status": "error", "error": "Empty query"}
    # Basic guardrail: limit query length
    if len(query) > 2000:
        query = query[:2000]
    print(f"  Query: {query[:100]}...")
    return {
        "query": query,
        "status": "planning",
        "messages": [f"[Ingest] Query received: {query[:80]}..."],
    }


def planner_node(state: ResearchState) -> Dict[str, Any]:
    """Decompose the query into sub-questions using LLM."""
    print("\n[Node: Planner] Decomposing query into sub-questions...")
    llm = create_llm(temperature=0.1)

    # FIX: Get the current date so the LLM knows what "latest" means
    current_date = datetime.now().strftime("%B %Y")

    system_prompt = f"""You are an expert research planner. Your job is to break down a user's research query into atomic sub-questions that can be answered through web search.

    Today's date is {current_date}.
    CRITICAL INSTRUCTION: Do NOT append "as of 2023" or past years to sub-questions. If the user asks for "latest" or "recent" information, formulate the search queries to target the current year ({current_date}) or recent months.

    For each sub-question, specify:
    1. A unique ID (e.g., "sq1", "sq2", etc.)
    2. A clear, searchable question
    3. The search strategy: "general" for broad web search, "news" for recent events, "academic" for research papers
    4. What type of evidence is expected
    5. Any dependencies on other sub-questions (by ID)

    Guidelines:
    - Make sub-questions specific and searchable
    - Aim for 3-5 sub-questions for most queries
    - Include both factual and analytical sub-questions
    - If the query is comparative, include sub-questions for each entity
    - If the query is temporal, include sub-questions for each time period
    - Set status to "pending" for all sub-questions"""

    user_prompt = f"Research query to decompose:\n\n{state['query']}"

    result = call_llm_structured(llm, system_prompt, user_prompt, ResearchPlan)

    if result is None or not result.sub_questions:
        # Fallback: create a single sub-question from the original query
        print("  [Planner] Fallback: using original query as single sub-question")
        plan = [SubQuestion(
            id="sq1",
            text=state["query"],
            search_strategy="general",
            expected_evidence="general information",
            dependencies=[],
            status=SubQuestionStatus.PENDING,
        )]
    else:
        plan = result.sub_questions
        print(f"  [Planner] Created {len(plan)} sub-questions:")
        for sq in plan:
            print(f"    - {sq.id}: {sq.text[:80]}...")

    return {
        "plan": plan,
        "status": "searching",
        "messages": [f"[Planner] Created {len(plan)} sub-questions"],
    }


def select_subquestion_node(state: ResearchState) -> Dict[str, Any]:
    """Find the next pending sub-question whose dependencies are met."""
    print("\n[Node: Select] Looking for next sub-question to research...")
    plan = state["plan"]

    # Find done sub-question IDs
    done_ids = {
        sq.id for sq in plan
        if sq.status in (SubQuestionStatus.DONE, SubQuestionStatus.SKIPPED)
    }

    # Find next pending sub-question with met dependencies
    next_sq = None
    for sq in plan:
        if sq.status == SubQuestionStatus.PENDING:
            if all(dep in done_ids for dep in sq.dependencies):
                next_sq = sq
                break

    if next_sq is None:
        # No more pending sub-questions
        print("  [Select] No more pending sub-questions. Moving to reflection.")
        return {
            "current_sq_id": None,
            "status": "reflecting",
            "messages": ["[Select] No more pending sub-questions"],
        }

    # Mark as in_progress
    next_sq.status = SubQuestionStatus.IN_PROGRESS
    print(f"  [Select] Selected: {next_sq.id} - {next_sq.text[:80]}...")

    return {
        "plan": plan,
        "current_sq_id": next_sq.id,
        "status": "searching",
        "raw_search_results": [],  # clear previous results
        "fetched_pages": [],  # clear previous pages
        "messages": [f"[Select] Selected sub-question: {next_sq.id}"],
    }


def search_node(state: ResearchState) -> Dict[str, Any]:
    """Execute web search for the current sub-question."""
    current_sq_id = state.get("current_sq_id")
    if not current_sq_id:
        return {"raw_search_results": []}

    plan = state["plan"]
    current_sq = next((sq for sq in plan if sq.id == current_sq_id), None)
    if not current_sq:
        return {"raw_search_results": []}

    print(f"\n[Node: Search] Searching for: {current_sq.text[:80]}...")
    print(f"  Strategy: {current_sq.search_strategy}")

    provider = get_search_provider()
    
    # FIX: Pass the search strategy to the provider
    results = provider.search(
        current_sq.text, 
        max_results=MAX_SEARCH_RESULTS, 
        strategy=current_sq.search_strategy
    )

    print(f"  [Search] Found {len(results)} results")
    for i, r in enumerate(results[:3]):
        print(f"    {i+1}. {r.get('title', 'No title')[:60]}...")
        print(f"       {r.get('url', '')[:80]}")

    return {
        "raw_search_results": results,
        "messages": [f"[Search] Found {len(results)} results for {current_sq_id}"],
    }


def fetch_pages_node(state: ResearchState) -> Dict[str, Any]:
    """Fetch full page content for top search results."""
    search_results = state.get("raw_search_results", [])
    if not search_results:
        return {"fetched_pages": []}

    print(f"\n[Node: Fetch] Fetching top {MAX_PAGES_PER_SUBQUESTION} pages...")
    fetched = []
    for result in search_results[:MAX_PAGES_PER_SUBQUESTION]:
        url = result.get("url", "")
        if not url:
            continue
        print(f"  [Fetch] Getting: {url[:80]}...")
        content = fetch_page_content(url)
        if content:
            fetched.append({
                "url": url,
                "title": result.get("title", ""),
                "content": content,
                "snippet": result.get("snippet", ""),
            })
            print(f"    -> {len(content)} chars retrieved")
        else:
            # Use snippet as fallback
            if result.get("snippet"):
                fetched.append({
                    "url": url,
                    "title": result.get("title", ""),
                    "content": result["snippet"],
                    "snippet": result["snippet"],
                })
                print(f"    -> Using snippet ({len(result['snippet'])} chars)")
        time.sleep(0.5)  # Rate limiting

    print(f"  [Fetch] Fetched {len(fetched)} pages")
    return {
        "fetched_pages": fetched,
        "messages": [f"[Fetch] Fetched {len(fetched)} pages"],
    }


def extract_evidence_node(state: ResearchState) -> Dict[str, Any]:
    """Extract relevant evidence from fetched pages using LLM."""
    current_sq_id = state.get("current_sq_id")
    plan = state["plan"]
    
    # FIX: Ensure we always find the sub-question and handle missing cases safely
    current_sq = next((sq for sq in plan if sq.id == current_sq_id), None)
    
    if not current_sq:
        print(f"  [Extract] Error: Could not find sub-question {current_sq_id} in plan!")
        return {
            "evidence": [],
            "messages": [f"[Extract] Error: sub-question {current_sq_id} not found"],
        }

    fetched_pages = state.get("fetched_pages", [])
    if not fetched_pages:
        # Mark sub-question as done with no evidence
        current_sq.status = SubQuestionStatus.DONE
        return {
            "plan": plan,
            "evidence": [],
            "messages": [f"[Extract] No pages to extract for {current_sq_id}"],
        }

    print(f"\n[Node: Extract] Extracting evidence for: {current_sq.text[:60]}...")
    llm = create_llm(temperature=0.0)

    system_prompt = """You are an expert evidence extractor. Given web page content and a research question, extract the most relevant passages that help answer the question.

For each piece of evidence:
1. State the factual claim or finding clearly
2. Provide the exact quote or close paraphrase
3. Rate relevance from 0.0 to 1.0 (how directly does it answer the question?)
4. Rate credibility from 0.0 to 1.0 (based on source authority and recency)

Guidelines:
- Only extract evidence relevant to the specific sub-question
- Prefer quantitative data over qualitative statements
- Note any caveats mentioned in the source
- If the page doesn't contain relevant evidence, return an empty list
- Do not make up information - only extract what's actually in the text"""

    all_evidence = []
    for page in fetched_pages:
        page_content = page["content"][:MAX_PAGE_TOKENS]
        user_prompt = f"""Sub-question to answer: {current_sq.text}

Page title: {page.get('title', 'Unknown')}
Page URL: {page.get('url', '')}
Page content:
---
{page_content}
---

Extract evidence from this page that helps answer the sub-question."""

        result = call_llm_structured(llm, system_prompt, user_prompt, ExtractedEvidence)

        if result and result.evidence_items:
            for item in result.evidence_items:
                item.sub_question_id = current_sq_id
                item.source_url = page.get("url", item.source_url)
                item.source_title = page.get("title", item.source_title)
                all_evidence.append(item)
                print(f"  [Extract] Found: {item.claim[:60]}... (rel={item.relevance}, cred={item.credibility})")

    # FIX: Always mark as DONE to prevent infinite loops
    current_sq.status = SubQuestionStatus.DONE
    print(f"  [Extract] Total evidence items: {len(all_evidence)}")

    return {
        "plan": plan,
        "evidence": all_evidence,
        "messages": [f"[Extract] Extracted {len(all_evidence)} evidence items for {current_sq_id}"],
    }


def reflect_node(state: ResearchState) -> Dict[str, Any]:
    """Evaluate whether collected evidence is sufficient."""
    print("\n[Node: Reflect] Evaluating evidence sufficiency...")
    iteration = state.get("iteration", 0) + 1

    evidence = state.get("evidence", [])
    plan = state["plan"]
    print(f"  [Reflect] Iteration {iteration}/{state.get('max_iterations', MAX_ITERATIONS)}")
    print(f"  [Reflect] Total evidence items: {len(evidence)}")

    # FIX 1: Check max_iterations FIRST to prevent infinite loops
    if iteration >= state.get("max_iterations", MAX_ITERATIONS):
        print(f"  [Reflect] Max iterations reached, forcing synthesis")
        return {
            "iteration": iteration,
            "status": "done",
            "gaps": [],
            "messages": ["[Reflect] Max iterations, forcing synthesis"],
        }

    # FIX 2: Clean up any stuck IN_PROGRESS sub-questions
    # If a node failed to mark them DONE, we mark them DONE here so the graph can proceed.
    for sq in plan:
        if sq.status == SubQuestionStatus.IN_PROGRESS:
            sq.status = SubQuestionStatus.DONE
            print(f"  [Reflect] Cleaned up stuck IN_PROGRESS sub-question: {sq.id}")

    # Check if there are still PENDING sub-questions to research
    pending = [sq for sq in plan if sq.status == SubQuestionStatus.PENDING]
    if pending:
        print(f"  [Reflect] Still have {len(pending)} pending sub-questions.")
        return {
            "iteration": iteration,
            "status": "searching",
            "messages": ["[Reflect] Sub-questions still pending"],
        }

    llm = create_llm(temperature=0.1)

    system_prompt = """You are a research reflector. Evaluate whether the collected evidence is sufficient to comprehensively answer the user's original query.

Consider:
1. Are all aspects of the query addressed?
2. Is evidence corroborated by multiple sources?
3. Are there contradictions that need resolution?
4. Are there obvious gaps in the research?

If the evidence is insufficient, identify specific gaps as new sub-questions that should be researched. These should be atomic and searchable. Assign them IDs like "sq_gap_1", "sq_gap_2", etc., set status to "pending", and use "general" search strategy.

Rate your confidence in the current evidence (0.0 to 1.0)."""

    # Format evidence for the LLM
    evidence_text = ""
    for i, ev in enumerate(evidence):
        evidence_text += f"\n[Evidence {i+1}]\n"
        evidence_text += f"  Claim: {ev.claim}\n"
        evidence_text += f"  Source: {ev.source_url}\n"
        evidence_text += f"  Relevance: {ev.relevance}\n"
        evidence_text += f"  Credibility: {ev.credibility}\n"
        evidence_text += f"  Sub-question: {ev.sub_question_id}\n"

    plan_text = "\n".join(f"  - {sq.id}: {sq.text} [{sq.status.value}]" for sq in plan)

    user_prompt = f"""Original query: {state['query']}

Research plan:
{plan_text}

Collected evidence ({len(evidence)} items):
{evidence_text if evidence else "(No evidence collected yet)"}

Evaluate whether this evidence is sufficient to answer the original query."""

    result = call_llm_structured(llm, system_prompt, user_prompt, ReflectionResult)

    if result is None:
        # If reflection fails, assume sufficient to avoid infinite loop
        print("  [Reflect] Reflection failed, proceeding to synthesis")
        return {
            "iteration": iteration,
            "status": "done",
            "messages": ["[Reflect] Failed, forcing synthesis"],
        }

    print(f"  [Reflect] Sufficient: {result.is_sufficient}")
    print(f"  [Reflect] Confidence: {result.confidence}")
    print(f"  [Reflect] Gaps: {len(result.gaps)}")

    if result.is_sufficient:
        return {
            "iteration": iteration,
            "status": "done",
            "gaps": [],
            "messages": [f"[Reflect] Sufficient (confidence={result.confidence})"],
        }

    # Add new sub-questions to plan
    if result.new_sub_questions:
        for sq in result.new_sub_questions:
            sq.status = SubQuestionStatus.PENDING
        updated_plan = plan + result.new_sub_questions
        print(f"  [Reflect] Added {len(result.new_sub_questions)} new sub-questions")
    else:
        # Create sub-questions from gaps text
        updated_plan = plan
        for i, gap in enumerate(result.gaps):
            sq = SubQuestion(
                id=f"sq_gap_{i+1}",
                text=gap,
                search_strategy="general",
                expected_evidence="general information",
                dependencies=[],
                status=SubQuestionStatus.PENDING,
            )
            updated_plan.append(sq)
            print(f"  [Reflect] Created gap sub-question: {sq.id}")

    return {
        "plan": updated_plan,
        "iteration": iteration,
        "status": "searching",
        "gaps": result.gaps,
        "messages": [f"[Reflect] Gaps found, re-planning (iteration {iteration})"],
    }


def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """Compose final answer with citations."""
    print("\n[Node: Synthesize] Composing final answer...")
    
    # FIX: Increased max_tokens to allow for a much longer, detailed output
    llm = create_llm(temperature=0.2, max_tokens=8000)

    evidence = state.get("evidence", [])
    gaps = state.get("gaps", [])
    iteration = state.get("iteration", 0)

    system_prompt = """You are an expert research synthesizer. Compose a highly detailed, long-form, comprehensive research report answering the user's query using ONLY the provided evidence.

Guidelines:
- Write a minimum of 4-6 detailed paragraphs.
- Use Markdown formatting with clear headings (##) and subheadings (###) to structure the report logically.
- Include inline citations as [1], [2], etc., where each number corresponds to the evidence item.
- Do not summarize too briefly; expand on the findings, provide context, and explain the significance of the data.
- If evidence is insufficient for part of the answer, state so explicitly and explain what is missing.
- Acknowledge uncertainties and conflicting evidence.
- If max iterations were reached without sufficient evidence, note the caveats.
- Rate your overall confidence in the answer (0.0 to 1.0).
- Suggest 1-3 follow-up questions for further research.
- List all source URLs in the citations field."""

    # Format evidence with citation numbers
    evidence_text = ""
    citations = []
    for i, ev in enumerate(evidence):
        citation_id = f"[{i+1}]"
        evidence_text += f"\n{citation_id} Claim: {ev.claim}\n"
        evidence_text += f"   Source: {ev.source_url}\n"
        evidence_text += f"   Title: {ev.source_title}\n"
        evidence_text += f"   Relevance: {ev.relevance}, Credibility: {ev.credibility}\n"
        evidence_text += f"   Quote: {ev.snippet[:200]}...\n"
        citations.append({
            "citation_id": citation_id,
            "url": ev.source_url,
            "title": ev.source_title,
        })

    plan_text = "\n".join(f"  - {sq.id}: {sq.text}" for sq in state["plan"])

    caveats = ""
    if gaps:
        caveats = f"Research gaps identified: {'; '.join(gaps)}"
    if iteration >= state.get("max_iterations", MAX_ITERATIONS):
        caveats += " Maximum research iterations reached; answer may be incomplete."

    user_prompt = f"""Original query: {state['query']}

Research plan:
{plan_text}

Collected evidence ({len(evidence)} items):
{evidence_text if evidence else "(No evidence was collected)"}

Known gaps: {'; '.join(gaps) if gaps else 'None'}
Iterations completed: {iteration}

Synthesize a highly detailed, comprehensive answer with citations."""

    result = call_llm_structured(llm, system_prompt, user_prompt, FinalAnswer)

    if result is None:
        # Fallback: generate text answer
        print("  [Synthesize] Structured output failed, using text fallback")
        text_answer = call_llm_text(llm, system_prompt, user_prompt)
        result = FinalAnswer(
            answer=text_answer,
            citations=citations,
            confidence=0.3,
            follow_up_questions=[],
            caveats="Fallback synthesis used due to structured output failure.",
        )

    print(f"\n  [Synthesize] Answer length: {len(result.answer)} chars")
    print(f"  [Synthesize] Citations: {len(result.citations)}")
    print(f"  [Synthesize] Confidence: {result.confidence}")

    return {
        "final_answer": result,
        "status": "done",
        "messages": [f"[Synthesize] Final answer composed (confidence={result.confidence})"],
    }


# ============================================================
# 8. ROUTING FUNCTIONS
# ============================================================

def route_after_select(state: ResearchState) -> str:
    """Route after sub-question selection."""
    if state.get("status") == "reflecting":
        return "reflect"
    if state.get("current_sq_id"):
        return "search"
    return "reflect"


def route_after_extract(state: ResearchState) -> str:
    """Route after evidence extraction - always go back to select."""
    return "select"


def route_after_reflect(state: ResearchState) -> str:
    """Route after reflection."""
    status = state.get("status", "")
    if status == "done":
        return "synthesize"
    if status == "searching":
        return "select"
    return "synthesize"


# ============================================================
# 9. GRAPH CONSTRUCTION
# ============================================================

def build_research_graph():
    """Build and compile the LangGraph research agent."""
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("ingest", ingest_node)
    graph.add_node("plan", planner_node)
    graph.add_node("select", select_subquestion_node)
    graph.add_node("search", search_node)
    graph.add_node("fetch", fetch_pages_node)
    graph.add_node("extract", extract_evidence_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("synthesize", synthesize_node)

    # Add edges
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "plan")
    graph.add_edge("plan", "select")

    # Conditional routing after select
    graph.add_conditional_edges(
        "select",
        route_after_select,
        {"search": "search", "reflect": "reflect"},
    )

    # Search -> Fetch -> Extract -> Select (inner loop)
    graph.add_edge("search", "fetch")
    graph.add_edge("fetch", "extract")
    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {"select": "select"},
    )

    # Conditional routing after reflect
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"select": "select", "synthesize": "synthesize"},
    )

    # Synthesize -> END
    graph.add_edge("synthesize", END)

    # Compile with memory checkpointing
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    return compiled


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def run_research(query: str, thread_id: str = "research_session_1") -> FinalAnswer:
    """Run the research agent on a query."""
    print(f"\n{'='*60}")
    print(f"RESEARCH AGENT STARTING")
    print(f"Query: {query}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"{'='*60}")

    graph = build_research_graph()

    initial_state = {
        "query": query,
        "plan": [],
        "current_sq_id": None,
        "evidence": [],
        "gaps": [],
        "raw_search_results": [],
        "fetched_pages": [],
        "iteration": 0,
        "max_iterations": MAX_ITERATIONS,
        "status": "planning",
        "error": None,
        "final_answer": None,
        "messages": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    final_state = graph.invoke(initial_state, config=config)

    answer = final_state.get("final_answer")
    if answer is None:
        return FinalAnswer(
            answer="Research failed to produce an answer.",
            citations=[],
            confidence=0.0,
            follow_up_questions=[],
            caveats="Agent did not reach synthesis.",
        )

    return answer

