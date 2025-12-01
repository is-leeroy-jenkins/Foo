# example_mixed.py
import os
from _init__ import Fetch

def run_mixed() -> None:
    """
    Purpose:
        Blend DB facts with policy guidance, then ask for a combined note that
        references past turns (tests memory).
    Parameters:
        None
    Returns:
        None
    """
    os.environ["OPENAI_API_KEY"] = "<your_api_key>"

    fetch = Fetch(
        db_uri="sqlite:///./agency.db",
        doc_paths=["./guides/Travel.pdf", "./guides/Procurement.txt"],
        model="gpt-4o-mini",
        temperature=0.3,
    )

    print(fetch.query("What were Q2 obligations by program element? Return a tidy table."))
    print(fetch.query("From the travel guide, when is pre-approval required? Include any thresholds."))
    print(fetch.query("Draft a short CFO note combining both answers with key numbers and policy constraints."))

    print("\n--- Chat History ---")
    for line in fetch.chat_history() or []:
        print(line)

if __name__ == "__main__":
    run_mixed()
