# example_docs.py
import os
from _init__ import Fetch

def run_docs() -> None:
    """
    Purpose:
        Answer questions grounded in uploaded documents (PDF/TXT/CSV/HTML).
    Parameters:
        None
    Returns:
        None
    """
    os.environ["OPENAI_API_KEY"] = "<your_api_key>"

    fetch = Fetch(
        db_uri="sqlite:///./empty.db",
        doc_paths=[
            "./policies/TravelPolicy.pdf",
            "./notes/Appropriations.txt",
            "./site/Acquisitions.html"
        ],
        model="gpt-4o-mini",
        temperature=0.3,
    )

    print(fetch.query("What approvals are required for international travel?"))
    print(fetch.query("Summarize micro-purchase rules in 3 bullets."))

if __name__ == "__main__":
    run_docs()
