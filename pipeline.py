from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain
from database import save_report
from vector_store import save_report as save_report_to_vectorstore


def run_research_pipeline(topic: str) -> dict:

    state = {}

    #search agent working
    print("\n"+ "="*50)
    print("step 1 - search agent is working...")
    print("="*50 )

    search_agent = build_search_agent()
    search_result = search_agent.invoke({

        "messages" : [("user", f"find recent and reliable information about: {topic}")]

    })

    state['search_results'] = search_result['messages'] [-1].content
    print("\nSearch result ",state['search_results'])

    
    #step 2 - reader agent 
    print("\n"+ "="*50)
    print("step 2 - reader agent is scraping top resources...")
    print("="*50 )

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

    #step 3 - writer chain

    print("\n"+ "="*50)
    print("step 3 - writer chain is drafting the report...")
    print("="*50 )

    # NOTE: kept identical to app.py — no truncation, so report quality matches the UI.
    research_combined = (
    f"Search Results:\n{state['search_results']}\n\n"
    f"Details Scraped Content:\n{state['scraped_content']}"
    )

    writer_result = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    state["report"] = writer_result
    print("\n Final Report: \n", state['report'])

    # Index into the vector store (tagged with topic) — mirrors app.py
    save_report_to_vectorstore(state["report"], topic)


    #critic report

    print("\n"+ "="*50)
    print("step 4 - critic chain is reviewing the report...")
    print("="*50 )

    state["feedback"] = critic_chain.invoke({
        "report": state['report']
    })
    print("\n critic report \n", state['feedback'])

    # Persist to SQLite history — mirrors app.py
    save_report(topic, state["report"], state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\n enter a research topic: ")
    run_research_pipeline(topic)