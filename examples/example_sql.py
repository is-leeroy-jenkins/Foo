# example_sql.py
import os
from _init__ import Fetch


def run_sql( ) -> None:
    """
    Purpose:
        Demonstrate structured queries via the SQL toolkit.
    Parameters:
        None
    Returns:
        None
    """
    os.environ[ "OPENAI_API_KEY" ] = "<your_api_key>"
    
    fetch = Fetch( db_uri="sqlite:///./finance.db",  doc_paths=[ ], model="gpt-4o-mini",
        temperature=0.2, )
    
    q1 = ("List the top 10 vendors by total obligations since Oct 1, 2023. "
          "Return a table (vendor, total_obligations) sorted desc.")
    print( fetch.query( q1 ) )
    
    q2 = (
        "Compute monthly obligations vs. outlays for FY 2024 and explain the variance in 3 bullets.")
    print( fetch.query( q2 ) )


if __name__ == "__main__":
    run_sql( )
