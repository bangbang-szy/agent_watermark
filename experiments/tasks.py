TASKS = {
    "qa": [
        "Search for the current homepage of LangGraph and summarize what it is used for in one sentence.",
        "Find one public reference about DuckDuckGo search and save a short note to notes/search.txt.",
    ],
    "database": [
        "Query the local database for the paper with the most citations and report its title and year.",
        "Compare average success_rate by framework in the agents table.",
    ],
    "reasoning": [
        "Use Python to compute the mean citations of papers after 2023 from the database output.",
        "Write a small file containing the names of available agent frameworks from the database.",
    ],
    "tool_chaining": [
        "Query the database for watermarking papers, compute total citations in Python, then answer.",
        "Read notes/search.txt if it exists, otherwise search for LangChain and write the note first.",
    ],
    "robust_workflows": [
        "Search for LangGraph, write a two sentence note to notes/langgraph.txt, then read it back and answer with the first sentence.",
        "Query all papers from the database, use Python to compute citation totals by area, and report the highest area.",
        "Query the agents table, save the rows as notes/agents.json, then answer which framework has the highest success_rate.",
        "Read notes/agents.json if present; otherwise query the database first. Then summarize the number of frameworks.",
    ],
    "mixed_decision": [
        "Decide whether the local database or web search is better for finding the success_rate of graph-researcher, then use the right tool.",
        "Use the database to find papers after 2023, then use Python to sort them by citations and answer with the top two titles.",
        "Search for a short description of SQLite, then compare it with the local SQLite database use in this project.",
        "Create notes/watermark_eval.txt containing the names of the available tools used by this agent.",
    ],
}


def all_tasks():
    return [task for group in TASKS.values() for task in group]
