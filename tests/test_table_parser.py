import pandas as pd
from src.ingestion.table_parser import clean_sec_table

def test_clean_sec_table_alignment_and_symbols():
    """
    Tests the advanced cleaning logic:
    1. Top-left cell padding ("Metric/Region")
    2. Column deduplication ("2025" & "2025")
    3. Symbol merging ("$" + "100" -> "$100")
    4. Phantom floats ("100" & "100.0" -> "$100.0")
    5. Percent merging ("5" + "%" -> "5%")
    """
    
    # 1. Create a deliberately messy "mock" SEC table
    messy_data = {
        0: ["", "Americas"],             # Missing top-left header
        1: ["2025", "$"],                # Disconnected dollar sign
        2: ["2025", "100"],              # Duplicated year
        3: ["Change", "100.0"],          # Phantom float
        4: ["Change", "5"],              # Duplicated header
        5: ["", "%"]                     # Disconnected percent sign
    }
    
    df = pd.DataFrame(messy_data)
    
    # 2. Define exactly what the perfect output SHOULD look like
    expected_markdown = (
        "Metric/Region | 2025 | Change\n"
        "Americas | $100 | 5%"
    )
    
    # 3. Run the function
    result = clean_sec_table(df)
    
    # 4. Assert that the result matches our expectation perfectly
    assert result == expected_markdown