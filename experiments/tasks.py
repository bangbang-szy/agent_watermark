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
}


def all_tasks():
    return [task for group in TASKS.values() for task in group]
