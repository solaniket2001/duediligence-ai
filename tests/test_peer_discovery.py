import pytest
from unittest.mock import patch, MagicMock
from src.agent.graph import discover_peer

@patch("src.agent.graph.get_llm")
def test_discover_peer_resolution(mock_get_llm):
    """
    Tests that the discover_peer node correctly parses different company 
    names into tickers, even if the LLM returns messy formats or dictionaries.
    """
    # Setup our fake LLM
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    # A list of test cases: (User Input Name, Simulated LLM Output, Expected Ticker)
    test_cases = [
        # Standard clean response
        ("apple", "AAPL", "AAPL"),
        
        # New Google SDK Dictionary Format (The bug we fixed earlier)
        ("microsoft", [{'TYPE': 'TEXT', 'text': 'MSFT', 'EXTRAS': {'SIGNATURE': 'xyz'}}], "MSFT"),
        
        # Chatty LLM response (Regex should isolate the ticker)
        ("amazon", "The ticker symbol for Amazon is AMZN.", "AMZN"),
        
        # Single letter ticker
        ("ford", "F", "F"),
        
        # Multi-word company names
        ("meta platforms", "META", "META"),
        
        # Lowercase LLM response
        ("alphabet", "googl", "GOOGL"),
    ]

    for peer_name, mock_response_content, expected_ticker in test_cases:
        # 1. Instruct the fake LLM to return our simulated content
        mock_response = MagicMock()
        mock_response.content = mock_response_content
        mock_llm.invoke.return_value = mock_response

        # 2. Setup the LangGraph state exactly as it looks entering the node
        state = {
            "company_name": "Tesla, Inc.", 
            "ticker": "TSLA",
            "peer_name": peer_name, 
            "peer_ticker": None
        }

        # 3. Run the node
        result = discover_peer(state)

        # 4. Assert the regex extractor correctly isolated the ticker
        assert result["peer_ticker"] == expected_ticker, f"Failed on {peer_name}"
        assert result["peer_name"] == peer_name