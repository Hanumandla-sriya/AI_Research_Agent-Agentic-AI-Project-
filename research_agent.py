"""
AI Research Agent using LangGraph + OpenRouter
==============================================
A stateful, multi-step research agent that decomposes queries,
searches the web, extracts evidence, reflects on sufficiency,
and synthesizes cited answers.

Environment variables:
    OPENROUTER_API_KEY   (required)
    OPENROUTER_MODEL     (recommended; copy an exact slug from openrouter.ai/models)
    TAVILY_API_KEY       (optional; DuckDuckGo is used if missing or if Tavily fails)
"""

import os
import re
import time
import json
import operator
from typing import TypedDict, List, Optional, Literal, Annotated, Dict, Any
from datetime import datetime, timezone
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

def _env(name: str, default: str = "") -> str:
    """Read an env var and strip whitespace/quotes (common copy-paste problem)."""
    return os.environ.get(name, default).strip().strip('"').strip("'")


# Any OpenAI-compatible provider works (Gemini, Groq, OpenRouter, ...).
#   LLM_API_KEY   - the provider's API key   (falls back to OPENROUTER_API_KEY)
#   LLM_BASE_URL  - the provider's OpenAI-compatible endpoint (default: OpenRouter)
#   LLM_MODEL     - exact model name for that provider (falls back to OPENROUTER_MODEL)
LLM_API_KEY = _env("LLM_API_KEY") or _env("OPENROUTER_API_KEY")
LLM_BASE_URL = _env("LLM_BASE_URL") or "https://openrouter.ai/api/v1"
DEFAULT_MODEL = _env("LLM_MODEL") or _env("OPENROUTER_MODEL") or "anthropic/claude-sonnet-4.5"
# Backwards-compatible aliases (app.py may still reference these names)
OPENROUTER_API_KEY = LLM_API_KEY
OPENROUTER_BASE_URL = LLM_BASE_URL
TAVILY_API_KEY = _env("TAVILY_API_KEY")
LLM_MIN_INTERVAL = float(_env("LLM_MIN_INTERVAL", "4"))  # seconds between LLM calls (free tiers cap requests/minute)

MAX_ITERATIONS = 3          # research rounds (1 = single pass, no gap-filling loop)
MAX_SUBQUESTIONS = 4        # cap on the initial plan size (fewer = fewer LLM calls)
MAX_GAP_QUESTIONS = 3       # cap on new sub-questions per reflection round
MAX_SEARCH_RESULTS = 6
MAX_PAGES_PER_SUBQUESTION = 3
MAX_PAGE_TOKENS = 6000      # character budget per page
MIN_RELEVANCE = 0.4         # drop evidence below this relevance
MAX_EVIDENCE_FOR_SYNTHESIS = 30
RECURSION_LIMIT = 150       # LangGraph default (25) is too low for multi-sub-question runs


def create_llm(temperature: float = 0.1, max_tokens: int = 4000):
    """Create an LLM instance connected to OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("Set the LLM_API_KEY (or OPENROUTER_API_KEY) environment variable.")
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=OPENROUTER_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=1,
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
        default="general",
        description="Search strategy: 'general', 'news', 'academic', or 'domain'"
    )
    expected_evidence: str = Field(default="", description="What type of evidence is expected")
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of sub-questions that must be answered first (usually empty)"
    )
    status: SubQuestionStatus = Field(default=SubQuestionStatus.PENDING)


class ResearchPlan(BaseModel):
    """Structured plan output from the planner node."""
    sub_questions: List[SubQuestion] = Field(
        description="Ordered list of sub-questions to research"
    )
    reasoning: str = Field(default="", description="Why this decomposition was chosen")


class EvidenceItem(BaseModel):
    """A single piece of extracted evidence."""
    claim: str = Field(description="The factual claim or finding")
    source_url: str = Field(default="", description="URL of the source")
    source_title: str = Field(default="", description="Title of the source page")
    snippet: str = Field(default="", description="Exact quote or close paraphrase from source")
    relevance: float = Field(default=0.5, description="Relevance score 0-1")
    credibility: float = Field(default=0.5, description="Credibility score 0-1")
    sub_question_id: str = Field(default="", description="Which sub-question this evidence addresses")
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


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
    confidence: float = Field(default=0.5, description="Confidence in current evidence, 0-1")
    gaps: List[str] = Field(
        default_factory=list,
        description="Specific gaps as new sub-question texts (empty if sufficient)"
    )
    reasoning: str = Field(default="", description="Explanation of the assessment")
    new_sub_questions: List[SubQuestion] = Field(
        default_factory=list,
        description="New sub-questions to add to the plan (empty if sufficient)"
    )


class FinalAnswer(BaseModel):
    """Final answer with citations."""
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

class ResearchState(TypedDict, total=False):
    """Shared state across all nodes in the research graph."""
    query: str
    plan: List[SubQuestion]
    current_sq_id: Optional[str]
    evidence: Annotated[List[EvidenceItem], operator.add]
    gaps: List[str]
    raw_search_results: List[Dict[str, Any]]
    fetched_pages: List[Dict[str, str]]
    seen_urls: Annotated[List[str], operator.add]
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

    def search(
        self, query: str, max_results: int = MAX_SEARCH_RESULTS, strategy: str = "general"
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class TavilySearchProvider(SearchProvider):
    """Search provider using the Tavily API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.tavily.com/search"

    def search(
        self, query: str, max_results: int = MAX_SEARCH_RESULTS, strategy: str = "general"
    ) -> List[Dict[str, Any]]:
        headers = {
            "Content-Type": "application/json",
            # Current Tavily API expects Bearer auth; the body key is kept for older API versions.
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "topic": "news" if strategy == "news" else "general",
            "include_answer": False,
        }
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(self.endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            return [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "source_type": "web",
                }
                for r in data.get("results", [])
            ]
        except Exception as e:
            print(f"  [Tavily] Search error: {e}")
            return []


class DuckDuckGoSearchProvider(SearchProvider):
    """Search provider using DuckDuckGo (no API key needed)."""

    def search(
        self, query: str, max_results: int = MAX_SEARCH_RESULTS, strategy: str = "general"
    ) -> List[Dict[str, Any]]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                print("  [DuckDuckGo] Package not installed. Install with: pip install ddgs")
                return []

        def _run(kind: str) -> List[Dict[str, Any]]:
            out = []
            with DDGS() as ddgs:
                fn = ddgs.news if kind == "news" else ddgs.text
                for r in fn(query, max_results=max_results):
                    out.append({
                        "url": r.get("href") or r.get("url", ""),
                        "title": r.get("title", ""),
                        "snippet": r.get("body") or r.get("snippet", ""),
                        "source_type": "web",
                    })
            return out

        # Datacenter IPs (e.g. Render) get throttled often, so retry with a short backoff.
        kinds = ["news", "text"] if strategy == "news" else ["text"]
        for attempt in range(3):
            for kind in kinds:
                try:
                    results = _run(kind)
                    if results:
                        return results
                except Exception as e:
                    print(f"  [DuckDuckGo] {kind} search error (attempt {attempt + 1}): {e}")
            time.sleep(1.5 * (attempt + 1))
        return []


def get_search_provider() -> SearchProvider:
    """Select the best available search provider."""
    if TAVILY_API_KEY:
        print("[Setup] Using Tavily search provider (DuckDuckGo as fallback)")
        return TavilySearchProvider(TAVILY_API_KEY)
    print("[Setup] Using DuckDuckGo search provider (no API key required)")
    return DuckDuckGoSearchProvider()


# ============================================================
# 5. PAGE FETCHER (Jina Reader)
# ============================================================

def fetch_page_content(url: str, max_chars: int = MAX_PAGE_TOKENS) -> str:
    """Fetch clean markdown content from a URL using Jina Reader."""
    jina_url = f"https://r.jina.ai/{url}"
    headers = {"Accept": "text/markdown", "X-Return-Format": "markdown"}
    try:
        with httpx.Client(timeout=25.0, follow_redirects=True) as client:
            resp = client.get(jina_url, headers=headers)
            if resp.status_code == 200:
                return resp.text[:max_chars]
            print(f"  [Fetch] Jina returned {resp.status_code} for {url}")
            return ""
    except Exception as e:
        print(f"  [Fetch] Error fetching {url}: {e}")
        return ""


# ============================================================
# 6. LLM HELPER FUNCTIONS
# ============================================================

def _extract_json(text: str) -> Optional[str]:
    """Pull a JSON object out of a model response (handles code fences and chatter)."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return None


_last_llm_call = 0.0
LLM_ERRORS: List[str] = []   # quota / rate-limit errors seen during the current run


def _throttle() -> None:
    """Space out LLM calls so free-tier requests-per-minute limits are respected."""
    global _last_llm_call
    wait = LLM_MIN_INTERVAL - (time.time() - _last_llm_call)
    if wait > 0:
        time.sleep(wait)
    _last_llm_call = time.time()


def _is_quota_error(e: Exception) -> bool:
    s_ = str(e)
    return ("429" in s_ or "rate limit" in s_.lower() or "quota" in s_.lower()
            or "RESOURCE_EXHAUSTED" in s_)


def _record_llm_error(e: Exception) -> None:
    msg = str(e)[:300]
    print(f"  [LLM] Rate limit / quota error: {msg}")
    LLM_ERRORS.append(msg)


def _failure_hint() -> str:
    if not LLM_ERRORS:
        return ""
    last = LLM_ERRORS[-1][:300]
    return ("\n\n**LLM error:** " + last +
            "\n\nFix: check LLM_API_KEY / LLM_BASE_URL / LLM_MODEL, wait if it is a rate limit, "
            "or increase LLM_MIN_INTERVAL.")


def call_llm_structured(
    llm: ChatOpenAI,
    system_prompt: str,
    user_prompt: str,
    output_model: type,
) -> Optional[BaseModel]:
    """Call LLM with structured output (tool calling), then fall back to plain JSON."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    # Attempt 1: function calling is the most widely supported structured mode.
    try:
        _throttle()
        structured_llm = llm.with_structured_output(output_model, method="function_calling")
        result = structured_llm.invoke(messages)
        if result is not None:
            return result
        raise ValueError("model returned no structured result")
    except Exception as e:
        if _is_quota_error(e):
            _record_llm_error(e)
            return None  # a JSON retry would only burn more quota
        print(f"  [LLM] Structured output failed ({str(e)[:200]}), trying JSON fallback...")

    # Attempt 2: ask for raw JSON and parse it.
    try:
        schema = output_model.model_json_schema()
        json_instruction = (
            "\n\nIMPORTANT: Respond with ONLY a valid JSON object (no prose, no code fences) "
            f"matching this schema:\n{json.dumps(schema, indent=2)}"
        )
        _throttle()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt + json_instruction),
        ])
        content = response.content if hasattr(response, "content") else str(response)
        raw = _extract_json(content)
        if raw:
            return output_model.model_validate_json(raw)
        print("  [LLM] JSON fallback: no JSON object found in response")
        LLM_ERRORS.append("Model did not return valid structured output.")
    except Exception as e2:
        if _is_quota_error(e2):
            _record_llm_error(e2)
        else:
            print(f"  [LLM] JSON fallback also failed: {str(e2)[:200]}")
            LLM_ERRORS.append(str(e2)[:300])
    return None


def call_llm_text(llm: ChatOpenAI, system_prompt: str, user_prompt: str) -> str:
    """Call LLM and return raw text."""
    try:
        _throttle()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        if _is_quota_error(e):
            _record_llm_error(e)
        else:
            print(f"  [LLM] Text call failed: {str(e)[:200]}")
            LLM_ERRORS.append(str(e)[:300])
        return ""


def to_search_query(text: str) -> str:
    """Condense a sub-question into a short keyword query (no LLM call, saves quota)."""
    q = re.sub(r"\([^)]*\)", " ", text)                      # drop parentheticals
    q = re.sub(r"\bas of\b[^?.,;]*", " ", q, flags=re.I)       # drop "as of <date>"
    q = re.sub(
        r"\b(what|how|why|which|who|are|is|do|does|can|could|should|the|of|in|and|to|an|be|"
        r"being|been|with|for|on|by|their|its|that|this|these|those|there|currently)\b",
        " ", q, flags=re.I,
    )
    q = re.sub(r"[^\w\s\-\+\.]", " ", q)
    words = q.split()[:12]
    cleaned = " ".join(words).strip()
    return cleaned or text[:150]


# ============================================================
# 7. NODE IMPLEMENTATIONS
# ============================================================

def ingest_node(state: ResearchState) -> Dict[str, Any]:
    """Normalize input and apply basic guardrails."""
    print(f"\n{'=' * 60}")
    print("[Node: Ingest] Processing user query...")
    query = (state.get("query") or "").strip()
    if not query:
        return {"status": "error", "error": "Empty query"}
    if len(query) > 2000:
        query = query[:2000]
    print(f"  Query: {query[:100]}")
    return {
        "query": query,
        "status": "planning",
        "messages": [f"[Ingest] Query received: {query[:80]}"],
    }


def _normalize_plan(plan: List[SubQuestion]) -> List[SubQuestion]:
    """Make an LLM-produced plan safe to execute (unique ids, pending status, valid deps)."""
    cleaned: List[SubQuestion] = []
    seen = set()
    for i, sq in enumerate(plan[:MAX_SUBQUESTIONS], start=1):
        if not sq.text or not sq.text.strip():
            continue
        sq_id = (sq.id or "").strip() or f"sq{i}"
        if sq_id in seen:
            sq_id = f"sq{i}_{len(seen)}"
        seen.add(sq_id)
        sq.id = sq_id
        sq.status = SubQuestionStatus.PENDING
        if sq.search_strategy not in ("general", "news", "academic", "domain"):
            sq.search_strategy = "general"
        cleaned.append(sq)
    valid_ids = {sq.id for sq in cleaned}
    for sq in cleaned:
        sq.dependencies = [d for d in sq.dependencies if d in valid_ids and d != sq.id]
    return cleaned


def planner_node(state: ResearchState) -> Dict[str, Any]:
    """Decompose the query into sub-questions using the LLM."""
    print("\n[Node: Planner] Decomposing query into sub-questions...")
    llm = create_llm(temperature=0.1)
    today = datetime.now().strftime("%B %d, %Y")

    system_prompt = f"""You are an expert research planner. Break the user's research query into atomic sub-questions that can be answered through web search.

Today's date is {today}.

Rules:
- Do NOT put dates, years, or phrases like "as of ..." inside the sub-question text. If recency matters, set the search strategy to "news" instead.
- Make each sub-question specific and searchable.
- Produce 3-5 sub-questions.
- Include both factual and analytical sub-questions.
- If the query is comparative, include a sub-question for each entity.
- Sub-questions should be independent: leave "dependencies" as an empty list unless truly required.
- Use ids "sq1", "sq2", ...
- Search strategy: "general" for broad search, "news" for recent events, "academic" for research papers.
- Set status to "pending" for all sub-questions."""

    user_prompt = f"Research query to decompose:\n\n{state['query']}"
    result = call_llm_structured(llm, system_prompt, user_prompt, ResearchPlan)

    plan: List[SubQuestion] = []
    if result is not None and result.sub_questions:
        plan = _normalize_plan(result.sub_questions)

    if not plan:
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
        print(f"  [Planner] Created {len(plan)} sub-questions:")
        for sq in plan:
            print(f"    - {sq.id} [{sq.search_strategy}]: {sq.text[:90]}")

    return {
        "plan": plan,
        "status": "searching",
        "messages": [f"[Planner] Created {len(plan)} sub-questions"],
    }


def select_subquestion_node(state: ResearchState) -> Dict[str, Any]:
    """Find the next pending sub-question whose dependencies are met."""
    print("\n[Node: Select] Looking for next sub-question to research...")
    plan = state["plan"]

    done_ids = {
        sq.id for sq in plan
        if sq.status in (SubQuestionStatus.DONE, SubQuestionStatus.SKIPPED)
    }
    pending = [sq for sq in plan if sq.status == SubQuestionStatus.PENDING]

    next_sq = next((sq for sq in pending if all(d in done_ids for d in sq.dependencies)), None)

    # Deadlock guard: pending items exist but all are blocked by dependencies.
    if next_sq is None and pending:
        print("  [Select] Dependency deadlock detected; running first pending sub-question anyway")
        next_sq = pending[0]
        next_sq.dependencies = []

    if next_sq is None:
        print("  [Select] No more pending sub-questions. Moving to reflection.")
        return {
            "current_sq_id": None,
            "status": "reflecting",
            "messages": ["[Select] No more pending sub-questions"],
        }

    next_sq.status = SubQuestionStatus.IN_PROGRESS
    print(f"  [Select] Selected: {next_sq.id} - {next_sq.text[:90]}")
    return {
        "plan": plan,
        "current_sq_id": next_sq.id,
        "status": "searching",
        "raw_search_results": [],
        "fetched_pages": [],
        "messages": [f"[Select] Selected sub-question: {next_sq.id}"],
    }


def search_node(state: ResearchState) -> Dict[str, Any]:
    """Execute web search for the current sub-question."""
    current_sq_id = state.get("current_sq_id")
    if not current_sq_id:
        return {"raw_search_results": []}

    current_sq = next((sq for sq in state["plan"] if sq.id == current_sq_id), None)
    if not current_sq:
        return {"raw_search_results": []}

    query = to_search_query(current_sq.text)
    print(f"\n[Node: Search] Sub-question: {current_sq.text[:90]}")
    print(f"  Search query: {query}")
    print(f"  Strategy: {current_sq.search_strategy}")

    provider = get_search_provider()
    results = provider.search(query, max_results=MAX_SEARCH_RESULTS, strategy=current_sq.search_strategy)

    # Fall back to DuckDuckGo if the primary provider (Tavily) returned nothing.
    if not results and not isinstance(provider, DuckDuckGoSearchProvider):
        print("  [Search] Primary provider returned nothing, falling back to DuckDuckGo")
        results = DuckDuckGoSearchProvider().search(
            query, max_results=MAX_SEARCH_RESULTS, strategy=current_sq.search_strategy
        )

    print(f"  [Search] Found {len(results)} results")
    for i, r in enumerate(results[:3]):
        print(f"    {i + 1}. {r.get('title', 'No title')[:60]}")
        print(f"       {r.get('url', '')[:80]}")

    return {
        "raw_search_results": results,
        "messages": [f"[Search] Found {len(results)} results for {current_sq_id}"],
    }


def fetch_pages_node(state: ResearchState) -> Dict[str, Any]:
    """Fetch full page content for the top (not yet seen) search results."""
    search_results = state.get("raw_search_results", [])
    if not search_results:
        return {"fetched_pages": []}

    seen = set(state.get("seen_urls", []))
    candidates = [r for r in search_results if r.get("url") and r["url"] not in seen]

    print(f"\n[Node: Fetch] Fetching up to {MAX_PAGES_PER_SUBQUESTION} pages...")
    fetched, new_urls = [], []
    for result in candidates[:MAX_PAGES_PER_SUBQUESTION]:
        url = result["url"]
        print(f"  [Fetch] Getting: {url[:80]}")
        content = fetch_page_content(url)
        if content:
            print(f"    -> {len(content)} chars retrieved")
        elif result.get("snippet"):
            content = result["snippet"]
            print(f"    -> Using search snippet ({len(content)} chars)")
        if content:
            fetched.append({
                "url": url,
                "title": result.get("title", ""),
                "content": content,
                "snippet": result.get("snippet", ""),
            })
            new_urls.append(url)
        time.sleep(0.5)  # rate limiting

    print(f"  [Fetch] Fetched {len(fetched)} pages")
    return {
        "fetched_pages": fetched,
        "seen_urls": new_urls,
        "messages": [f"[Fetch] Fetched {len(fetched)} pages"],
    }


def extract_evidence_node(state: ResearchState) -> Dict[str, Any]:
    """Extract relevant evidence from fetched pages using the LLM."""
    current_sq_id = state.get("current_sq_id")
    plan = state["plan"]
    current_sq = next((sq for sq in plan if sq.id == current_sq_id), None)

    if not current_sq:
        print(f"  [Extract] Error: sub-question {current_sq_id} not found in plan")
        return {"evidence": [], "messages": [f"[Extract] sub-question {current_sq_id} not found"]}

    fetched_pages = state.get("fetched_pages", [])
    if not fetched_pages:
        current_sq.status = SubQuestionStatus.DONE
        print(f"  [Extract] No pages for {current_sq_id}")
        return {
            "plan": plan,
            "evidence": [],
            "messages": [f"[Extract] No pages to extract for {current_sq_id}"],
        }

    print(f"\n[Node: Extract] Extracting evidence for: {current_sq.text[:70]}")
    llm = create_llm(temperature=0.0)

    system_prompt = """You are an expert evidence extractor. Given web page content and a research question, extract the passages that help answer the question.

For each piece of evidence:
1. State the factual claim or finding clearly
2. Provide a short exact quote or close paraphrase as the snippet
3. Rate relevance from 0.0 to 1.0 (how directly it answers the question)
4. Rate credibility from 0.0 to 1.0 (source authority and recency)

Guidelines:
- Only extract evidence relevant to the specific question
- If the page is off-topic (for example it is about a different subject with similar words), return an empty list
- Prefer quantitative data over vague statements
- Note caveats mentioned in the source
- Do not make anything up; only extract what is in the text"""

    pages = state.get("fetched_pages", [])
    per_page = max(1500, 9000 // max(1, len(pages)))
    url_titles = {p["url"]: p.get("title", "") for p in pages}

    sources_text = ""
    for i, page in enumerate(pages, start=1):
        sources_text += (
            f"\n=== SOURCE {i} ===\nTitle: {page.get('title', 'Unknown')}\n"
            f"URL: {page['url']}\nContent:\n{page['content'][:per_page]}\n"
        )

    user_prompt = f"""Question to answer: {current_sq.text}

Below are {len(pages)} sources. For every evidence item, set source_url to the exact URL of the source it came from.
{sources_text}

Extract evidence from these sources that helps answer the question."""

    all_evidence: List[EvidenceItem] = []
    result = call_llm_structured(llm, system_prompt, user_prompt, ExtractedEvidence)
    if result and result.evidence_items:
        skipped = 0
        for item in result.evidence_items:
            item.relevance = max(0.0, min(1.0, float(item.relevance)))
            item.credibility = max(0.0, min(1.0, float(item.credibility)))
            if item.relevance < MIN_RELEVANCE or not item.claim.strip():
                continue
            url = item.source_url
            if url not in url_titles:
                match = next((u for u in url_titles if url and (u in url or url in u)), None)
                if match:
                    url = match
                elif len(pages) == 1:
                    url = pages[0]["url"]
                else:
                    by_title = next((u for u, t in url_titles.items()
                                     if t and item.source_title and t.lower() in item.source_title.lower()), None)
                    url = by_title or pages[0]["url"]
            item.source_url = url
            item.source_title = url_titles[url]
            item.sub_question_id = current_sq_id
            all_evidence.append(item)
            print(f"  [Extract] Found: {item.claim[:70]} (rel={item.relevance}, cred={item.credibility})")
        if skipped:
            print(f"  [Extract] Skipped {skipped} items with no matching source URL")

    if not all_evidence:
        print("  [Extract] LLM extraction gave nothing; using search snippets as evidence")
        for page in pages:
            text = (page.get("snippet") or page.get("content", "")[:400]).strip()
            if not text:
                continue
            all_evidence.append(EvidenceItem(
                claim=text[:300],
                source_url=page["url"],
                source_title=page.get("title", ""),
                snippet=text[:300],
                relevance=0.5,
                credibility=0.5,
                sub_question_id=current_sq_id,
            ))

    current_sq.status = SubQuestionStatus.DONE  # always mark done to avoid infinite loops
    print(f"  [Extract] Evidence items for {current_sq_id}: {len(all_evidence)}")
    return {
        "plan": plan,
        "evidence": all_evidence,
        "messages": [f"[Extract] Extracted {len(all_evidence)} evidence items for {current_sq_id}"],
    }


def reflect_node(state: ResearchState) -> Dict[str, Any]:
    """Evaluate whether collected evidence is sufficient; add gap sub-questions if not."""
    print("\n[Node: Reflect] Evaluating evidence sufficiency...")
    iteration = state.get("iteration", 0) + 1
    max_iter = state.get("max_iterations", MAX_ITERATIONS)
    evidence = state.get("evidence", [])
    plan = state["plan"]

    print(f"  [Reflect] Iteration {iteration}/{max_iter}")
    print(f"  [Reflect] Total evidence items: {len(evidence)}")

    # Clean up any sub-questions stuck in progress.
    for sq in plan:
        if sq.status == SubQuestionStatus.IN_PROGRESS:
            sq.status = SubQuestionStatus.DONE

    if iteration >= max_iter:
        print("  [Reflect] Max iterations reached, moving to synthesis")
        return {"plan": plan, "iteration": iteration, "status": "done", "gaps": [],
                "messages": ["[Reflect] Max iterations reached"]}

    if not evidence:
        # Searching is broken or returned nothing; looping again would waste calls.
        print("  [Reflect] No evidence at all; stopping instead of looping")
        return {"plan": plan, "iteration": iteration, "status": "done", "gaps": [],
                "messages": ["[Reflect] No evidence collected"]}

    llm = create_llm(temperature=0.1)

    system_prompt = f"""You are a research reflector. Decide whether the collected evidence is sufficient to comprehensively answer the user's original query.

Consider:
1. Are all aspects of the query addressed?
2. Is the evidence corroborated by multiple sources?
3. Are there contradictions?
4. Are there obvious gaps?

If insufficient, list up to {MAX_GAP_QUESTIONS} gaps as new atomic, searchable sub-questions (no dates or "as of" phrases). Give them ids like "sq_gap_1", status "pending", strategy "general", and no dependencies.
Rate your confidence in the current evidence from 0.0 to 1.0."""

    evidence_text = ""
    for i, ev in enumerate(evidence[:MAX_EVIDENCE_FOR_SYNTHESIS], start=1):
        evidence_text += (
            f"\n[Evidence {i}] {ev.claim}\n"
            f"  Source: {ev.source_url} | Sub-question: {ev.sub_question_id} "
            f"| rel={ev.relevance} cred={ev.credibility}\n"
        )
    plan_text = "\n".join(f"  - {sq.id}: {sq.text} [{sq.status.value}]" for sq in plan)

    user_prompt = f"""Original query: {state['query']}

Research plan:
{plan_text}

Collected evidence ({len(evidence)} items):
{evidence_text}

Evaluate whether this evidence is sufficient to answer the original query."""

    result = call_llm_structured(llm, system_prompt, user_prompt, ReflectionResult)

    if result is None:
        print("  [Reflect] Reflection failed, proceeding to synthesis")
        return {"plan": plan, "iteration": iteration, "status": "done",
                "messages": ["[Reflect] Failed, proceeding to synthesis"]}

    print(f"  [Reflect] Sufficient: {result.is_sufficient} | Confidence: {result.confidence} "
          f"| Gaps: {len(result.gaps)}")

    if result.is_sufficient or (not result.gaps and not result.new_sub_questions):
        return {"plan": plan, "iteration": iteration, "status": "done", "gaps": [],
                "messages": [f"[Reflect] Sufficient (confidence={result.confidence})"]}

    existing_ids = {sq.id for sq in plan}
    new_items: List[SubQuestion] = []
    if result.new_sub_questions:
        for sq in result.new_sub_questions[:MAX_GAP_QUESTIONS]:
            sq.status = SubQuestionStatus.PENDING
            sq.dependencies = []
            if sq.id in existing_ids or not sq.id:
                sq.id = f"sq_gap_{iteration}_{len(new_items) + 1}"
            if sq.search_strategy not in ("general", "news", "academic", "domain"):
                sq.search_strategy = "general"
            existing_ids.add(sq.id)
            new_items.append(sq)
    else:
        for i, gap in enumerate(result.gaps[:MAX_GAP_QUESTIONS], start=1):
            new_items.append(SubQuestion(
                id=f"sq_gap_{iteration}_{i}",
                text=gap,
                search_strategy="general",
                expected_evidence="general information",
                dependencies=[],
                status=SubQuestionStatus.PENDING,
            ))

    print(f"  [Reflect] Added {len(new_items)} gap sub-questions")
    return {
        "plan": plan + new_items,
        "iteration": iteration,
        "status": "searching",
        "gaps": result.gaps,
        "messages": [f"[Reflect] Gaps found, re-planning (iteration {iteration})"],
    }


def _split_followups(text: str) -> List[str]:
    """Pull bullet items out of a trailing 'Follow-up' section, if present."""
    m = re.search(r"\n#+\s*[^\n]*follow[- ]?up[^\n]*\n(.*)$", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    items = []
    for line in m.group(1).splitlines():
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
        if line:
            items.append(line)
    return items[:3]


def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    """Compose the final cited answer."""
    print("\n[Node: Synthesize] Composing final answer...")
    evidence = list(state.get("evidence", []))
    gaps = state.get("gaps", [])
    iteration = state.get("iteration", 0)

    # Never write a report from nothing.
    if not evidence:
        print("  [Synthesize] No evidence; returning failure message without an LLM call")
        result = FinalAnswer(
            answer=(
                "**No sources could be retrieved, so no report was generated.**\n\n"
                "Check the Render logs for search errors (Tavily 401 or DuckDuckGo timeouts), "
                "confirm the API keys are set, and try again."
                + _failure_hint()
            ),
            citations=[],
            confidence=0.0,
            follow_up_questions=[],
            caveats="Zero evidence items were collected.",
        )
        return {"final_answer": result, "status": "done",
                "messages": ["[Synthesize] Skipped: no evidence"]}

    # Keep the strongest evidence.
    evidence.sort(key=lambda e: e.relevance * e.credibility, reverse=True)
    evidence = evidence[:MAX_EVIDENCE_FOR_SYNTHESIS]

    # Number sources by unique URL so citations stay consistent.
    source_numbers: Dict[str, int] = {}
    citations: List[Dict[str, str]] = []
    for ev in evidence:
        if ev.source_url not in source_numbers:
            source_numbers[ev.source_url] = len(source_numbers) + 1
            citations.append({
                "citation_id": f"[{source_numbers[ev.source_url]}]",
                "url": ev.source_url,
                "title": ev.source_title or ev.source_url,
            })

    evidence_text = ""
    for ev in evidence:
        n = source_numbers[ev.source_url]
        evidence_text += (
            f"\n[{n}] {ev.claim}\n"
            f"    Source: {ev.source_title} ({ev.source_url})\n"
            f"    Quote: {ev.snippet[:250]}\n"
        )

    plan_text = "\n".join(f"  - {sq.id}: {sq.text}" for sq in state["plan"])

    system_prompt = """You are an expert research synthesizer. Write a detailed, long-form research report answering the user's query using ONLY the provided evidence.

Guidelines:
- Use Markdown with a title, an executive summary, and clear ## and ### headings.
- Write at least 4-6 substantial paragraphs; explain significance and context, not just facts.
- Cite with inline numbers like [1], [2] that match the source numbers in the evidence list. Cite every factual claim.
- If evidence is thin or missing for part of the query, say so explicitly. Do not fill gaps from memory.
- Note conflicting evidence and uncertainty.
- End with a section titled "## Suggested Follow-up Questions" containing 1-3 bullet points.
- Do not include a references list; it is added separately."""

    user_prompt = f"""Original query: {state['query']}

Research plan:
{plan_text}

Evidence ({len(evidence)} items from {len(citations)} sources):
{evidence_text}

Known gaps: {'; '.join(gaps) if gaps else 'None'}

Write the report."""

    llm = create_llm(temperature=0.2, max_tokens=6000)
    answer_text = call_llm_text(llm, system_prompt, user_prompt).strip()

    if not answer_text:
        lines = [
            "# Evidence summary",
            "",
            "The language model could not write the full report, so here are the "
            "evidence items that were collected (numbers match the sources list).",
            "",
        ]
        for ev in evidence:
            lines.append(f"- {ev.claim} [{source_numbers[ev.source_url]}]")
        answer_text = "\n".join(lines) + _failure_hint()
        confidence = 0.1
    else:
        # Confidence heuristic from evidence quality and sub-question coverage.
        quality = sum(e.relevance * e.credibility for e in evidence) / len(evidence)
        covered = {e.sub_question_id for e in evidence}
        total_sq = max(1, len(state["plan"]))
        coverage = min(1.0, len(covered) / total_sq)
        confidence = round(max(0.05, min(0.95, quality * (0.5 + 0.5 * coverage))), 2)

    caveats = []
    if gaps:
        caveats.append(f"Research gaps identified: {'; '.join(gaps)}.")
    if iteration >= state.get("max_iterations", MAX_ITERATIONS):
        caveats.append("Maximum research iterations reached; the answer may be incomplete.")
    if len(citations) < 3:
        caveats.append("Fewer than three independent sources were found.")

    result = FinalAnswer(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        follow_up_questions=_split_followups(answer_text),
        caveats=" ".join(caveats),
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
    if state.get("status") == "reflecting" or not state.get("current_sq_id"):
        return "reflect"
    return "search"


def route_after_reflect(state: ResearchState) -> str:
    return "select" if state.get("status") == "searching" else "synthesize"


# ============================================================
# 9. GRAPH CONSTRUCTION
# ============================================================

def build_research_graph():
    """Build and compile the LangGraph research agent."""
    graph = StateGraph(ResearchState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("plan", planner_node)
    graph.add_node("select", select_subquestion_node)
    graph.add_node("search", search_node)
    graph.add_node("fetch", fetch_pages_node)
    graph.add_node("extract", extract_evidence_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("synthesize", synthesize_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "plan")
    graph.add_edge("plan", "select")

    graph.add_conditional_edges(
        "select", route_after_select, {"search": "search", "reflect": "reflect"}
    )

    # Inner loop: search -> fetch -> extract -> select
    graph.add_edge("search", "fetch")
    graph.add_edge("fetch", "extract")
    graph.add_edge("extract", "select")

    graph.add_conditional_edges(
        "reflect", route_after_reflect, {"select": "select", "synthesize": "synthesize"}
    )

    graph.add_edge("synthesize", END)

    return graph.compile(checkpointer=MemorySaver())


# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def build_initial_state(query: str, max_iterations: int = MAX_ITERATIONS) -> Dict[str, Any]:
    """Initial state for the graph (use this if app.py streams the graph itself)."""
    return {
        "query": query,
        "plan": [],
        "current_sq_id": None,
        "evidence": [],
        "gaps": [],
        "raw_search_results": [],
        "fetched_pages": [],
        "seen_urls": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "status": "planning",
        "error": None,
        "final_answer": None,
        "messages": [],
    }


def run_research(
    query: str,
    thread_id: str = "research_session_1",
    max_iterations: int = MAX_ITERATIONS,
) -> FinalAnswer:
    """Run the research agent on a query."""
    print(f"\n{'=' * 60}")
    print("RESEARCH AGENT STARTING")
    print(f"Query: {query}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"LLM endpoint: {LLM_BASE_URL}")
    LLM_ERRORS.clear()
    print(f"Max iterations: {max_iterations}")
    print(f"{'=' * 60}")

    graph = build_research_graph()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    final_state = graph.invoke(build_initial_state(query, max_iterations), config=config)

    answer = final_state.get("final_answer")
    if answer is None:
        return FinalAnswer(
            answer="Research failed to produce an answer.",
            citations=[],
            confidence=0.0,
            follow_up_questions=[],
            caveats=final_state.get("error") or "Agent did not reach synthesis.",
        )
    return answer


if __name__ == "__main__":
    out = run_research("Explain agentic AI, its applications, benefits, and risks.", max_iterations=2)
    print("\n" + out.answer)
    print("\nConfidence:", out.confidence)
    for c in out.citations:
        print(c["citation_id"], c["title"], c["url"])