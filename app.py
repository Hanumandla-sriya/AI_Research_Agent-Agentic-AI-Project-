# app.py

# 1. Load environment variables FIRST, before any other imports!
from dotenv import load_dotenv
load_dotenv()

# 2. Now import the rest of the libraries
import gradio as gr
import time
from research_agent import build_research_graph, SubQuestionStatus

def run_research_stream(query, max_iter):
    """Generator function to stream agent steps and final answer to the UI."""
    graph = build_research_graph()
    thread_id = f"ui_thread_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 150}
    
    initial_state = {
        "query": query,
        "plan": [],
        "current_sq_id": None,
        "evidence": [],
        "gaps": [],
        "raw_search_results": [],
        "fetched_pages": [],
        "iteration": 0,
        "max_iterations": max_iter,  # Use the value from the UI slider
        "status": "planning",
        "error": None,
        "final_answer": None,
        "messages": [],
    }

    # Yields: progress, answer, confidence, cites, follows, caveats, status
    yield 5, "", 0.0, "", "", "", "Starting research..."

    # Stream LangGraph updates using 'values' to get the full state every time
    for event in graph.stream(initial_state, config=config, stream_mode="values"):
        status = event.get("status", "searching")
        iteration = event.get("iteration", 0)
        plan = event.get("plan", [])
        
        # Calculate strictly increasing overall progress
        if status == "planning" or not plan:
            progress_val = 5
            status_msg = "Decomposing query into sub-questions..."
        elif status == "done" and event.get("final_answer"):
            progress_val = 100
            status_msg = "Synthesizing final answer..."
        else:
            # Calculate how much of the research is done
            done_count = sum(1 for sq in plan if sq.status == SubQuestionStatus.DONE)
            total_count = max(len(plan), 1)
            
            # 80% of the bar is for research, 10% for planning, 10% for synthesis
            iter_weight = (min(iteration, max_iter) / max_iter) * 80
            sq_weight = (done_count / total_count) * (80 / max_iter)
            progress_val = 10 + iter_weight + sq_weight
            
            # Clamp between 10 and 95
            progress_val = min(95, max(10, progress_val))
            
            if status == "reflecting":
                status_msg = "Reflecting on findings..."
            else:
                sq_id = event.get("current_sq_id")
                status_msg = f"Researching: {sq_id}" if sq_id else "Evaluating..."

        # Yield the 7 outputs
        yield progress_val, "", 0.0, "", "", "", status_msg

    # Retrieve final state after stream completes
    final_state = graph.get_state(config).values
    final_answer_obj = final_state.get("final_answer")

    if final_answer_obj:
        cites_md = "\n".join(
            f"{c['citation_id']} [{c.get('title', 'Source')}]({c.get('url', '')})" 
            for c in final_answer_obj.citations
        )
        follows_md = "\n".join(f"- {q}" for q in final_answer_obj.follow_up_questions)
        status_msg = f"Done! (Confidence: {final_answer_obj.confidence*100:.0f}%)"
        
        yield 100, final_answer_obj.answer, final_answer_obj.confidence, cites_md, follows_md, final_answer_obj.caveats, status_msg
    else:
        yield 100, "Failed to generate an answer.", 0.0, "", "", "Agent failed to reach synthesis.", "Error"

# ============================================================
# GRADIO UI LAYOUT
# ============================================================

with gr.Blocks() as demo:
    gr.Markdown("# 🔬 AI Research Agent")
    gr.Markdown("Enter a complex research query. The agent will decompose it, search the web, extract evidence, reflect on its findings, and synthesize a cited answer.")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Research Query", 
            placeholder="e.g., Compare the energy density of solid-state batteries shipping in 2024 EVs versus the projections made in 2021",
            scale=3
        )
        max_iter_slider = gr.Slider(
            minimum=1, 
            maximum=10, 
            value=3, 
            step=1, 
            label="Max Research Iterations", 
            scale=1,
            info="Higher = deeper research, but takes longer."
        )
        submit_btn = gr.Button("Start Research", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            progress_output = gr.Slider(
                label="Overall Progress", 
                minimum=0, 
                maximum=100, 
                step=1, 
                interactive=False
            )
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            gr.Markdown("### Final Output")
            answer_output = gr.Markdown(label="Final Answer")
            
            with gr.Accordion("Metadata & Citations", open=False):
                confidence_output = gr.Slider(label="Confidence", minimum=0, maximum=1, interactive=False)
                caveats_output = gr.Textbox(label="Caveats", interactive=False)
                cites_output = gr.Markdown(label="Sources")
                follows_output = gr.Markdown(label="Suggested Follow-up Questions")

    # Click event triggers the stream, passing both the query and the slider value
    submit_btn.click(
        fn=run_research_stream,
        inputs=[query_input, max_iter_slider],
        outputs=[progress_output, answer_output, confidence_output, cites_output, follows_output, caveats_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), 
        css="footer {visibility: hidden}"
    )