from researchmind.agents import build_reader_agent, build_search_agent, writer_chain, critic_chain, parse_score
from researchmind.agents import ResearchReport, CriticReview
from researchmind.database import save_report
from researchmind.vector_store import save_report as save_report_to_vectorstore


# ── Helper functions to convert Pydantic models to display strings ──

def report_to_string(report: ResearchReport) -> str:
    """Convert ResearchReport Pydantic model to markdown string."""
    if isinstance(report, ResearchReport):
        md = f"""# Research Report

## Introduction
{report.introduction}

## Key Findings
"""
        for i, finding in enumerate(report.key_findings, 1):
            md += f"- {finding}\n"
        
        md += f"""
## Conclusion
{report.conclusion}

## Sources
"""
        for source in report.sources:
            md += f"- {source}\n"
        return md
    return str(report)


def critic_to_string(critic: CriticReview) -> str:
    """Convert CriticReview Pydantic model to string."""
    if isinstance(critic, CriticReview):
        md = f"""Score: {critic.score}/10

Strengths:
"""
        for strength in critic.strengths:
            md += f"- {strength}\n"
        
        md += "\nAreas to Improve:\n"
        for area in critic.areas_to_improve:
            md += f"- {area}\n"
        
        md += f"\n\nVerdict: {critic.verdict}"
        return md
    return str(critic)


# Quality gate: keep researching until the critic scores the report at
# least MIN_SCORE. MAX_ATTEMPTS stops it looping forever (the free model
# can plateau and never reach the target on its own).
MIN_SCORE = 7
MAX_ATTEMPTS = 3


def run_research_pipeline(topic: str) -> dict:

    state = {}

    # ── Search + Reader run ONCE — the gathered research is reused on every
    #    rewrite. Only the writer + critic loop below.

    #search agent working
    print("\n" + "=" * 50)
    print("step 1 - search agent is working...")
    print("=" * 50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({

        "messages": [("user", f"find recent and reliable information about: {topic}")]

    })

    state['search_results'] = search_result['messages'][-1].content
    print("\nSearch result ", state['search_results'])

    #step 2 - reader agent
    print("\n" + "=" * 50)
    print("step 2 - reader agent is scraping top resources...")
    print("=" * 50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content
    print("\nScraped content: \n", state['scraped_content'])

    # NOTE: kept identical to app.py — no truncation, so report quality matches the UI.
    research_combined = (
        f"Search Results:\n{state['search_results']}\n\n"
        f"Details Scraped Content:\n{state['scraped_content']}"
    )

    # ── Write + critique loop: rewrite (using the SAME research) until the
    #    critic scores >= MIN_SCORE, capped at MAX_ATTEMPTS.
    previous_feedback = None   # critic feedback from the last attempt (if any)

    for attempt in range(1, MAX_ATTEMPTS + 1):

        print("\n" + "#" * 50)
        print(f"WRITE/CRITIQUE ATTEMPT {attempt} of {MAX_ATTEMPTS}")
        print("#" * 50)

        #step 3 - writer chain

        print("\n" + "=" * 50)
        print("step 3 - writer chain is drafting the report...")
        print("=" * 50)

        # On a retry, hand the previous critique to the writer so it can
        # fix the weak spots instead of repeating the same mistakes. The
        # underlying research stays the same — only the writing is redone.
        writer_input = research_combined
        if previous_feedback:
            # Convert CriticReview to string if needed
            feedback_text = critic_to_string(previous_feedback) if isinstance(previous_feedback, CriticReview) else str(previous_feedback)
            writer_input += (
                "\n\nPREVIOUS REPORT FEEDBACK (the last draft scored too low — "
                "fix every issue below to improve the report):\n"
                f"{feedback_text}"
            )

        writer_result = writer_chain.invoke({
            "topic": topic,
            "research": writer_input
        })

        state["report"] = writer_result
        print("\n Final Report: \n", report_to_string(state['report']))

        #critic report

        print("\n" + "=" * 50)
        print("step 4 - critic chain is reviewing the report...")
        print("=" * 50)

        state["feedback"] = critic_chain.invoke({
            "report": report_to_string(state['report'])
        })
        print("\n critic report \n", critic_to_string(state['feedback']))

        # Read the score back out of the critic's feedback.
        state["score"] = parse_score(state["feedback"])
        print(f"\n>>> Critic score: {state['score']} (need >= {MIN_SCORE})")

        # Accept the report if it hit the target (or if no score could be
        # parsed — better to stop than loop blindly).
        if state["score"] is None or state["score"] >= MIN_SCORE:
            print(">>> Report accepted.")
            break

        # Otherwise: rewrite the report, carrying this critique forward.
        print(f">>> Score {state['score']} <= 6 — rewriting the report...")
        previous_feedback = state["feedback"]

    else:
        # Loop finished without breaking = never reached MIN_SCORE.
        print(f"\n>>> Reached max {MAX_ATTEMPTS} attempts. "
              f"Keeping best effort (score {state.get('score')}).")

    state["attempts"] = attempt

    # Save only the final accepted/best report — once, after the loop.
    # Convert Pydantic models to strings
    report_str = report_to_string(state["report"])
    feedback_str = critic_to_string(state["feedback"])
    
    save_report_to_vectorstore(report_str, topic)
    save_report(topic, report_str, feedback_str)

    return state


if __name__ == "__main__":
    topic = input("\n enter a research topic: ")
    run_research_pipeline(topic)
