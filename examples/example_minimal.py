# example_minimal.py
import os
from _init__ import Fetch  # adjust to your actual module name

def main() -> None:
    """
    Purpose:
        Quick end-to-end test of Fetch. Works even if doc_paths is empty.
    Parameters:
        None
    Returns:
        None
    """
    os.environ["OPENAI_API_KEY"] = "<your_api_key>"

    fetch = Fetch(
        db_uri="sqlite:///./example.db",
        doc_paths=["./docs/handbook.pdf", "./docs/faq.txt"],  # or []
        model="gpt-4o-mini",
        temperature=0.3,
    )

    print(fetch.query("Say 'ready' if initialization succeeded."))

if __name__ == "__main__":
    main()
