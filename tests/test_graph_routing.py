import pytest
from src.agent.graph import should_refine_financials

def test_should_refine_financials_low_score():
    # If the score is below 0.80 and retries < 2, it should trigger a rewrite
    state = {
        "financial_score": 0.75, 
        "financial_retry_count": 0, 
        "financial_context": ["Some SEC text"]
    }
    assert should_refine_financials(state) == "rewrite_financial_query"

def test_should_refine_financials_high_score():
    # If the score is above 0.80, it should proceed to risk extraction
    state = {
        "financial_score": 0.95, 
        "financial_retry_count": 0, 
        "financial_context": ["Some SEC text"]
    }
    assert should_refine_financials(state) == "extract_risks"

def test_should_refine_financials_max_retries():
    # If the score is low but we hit the max retry limit, force it forward to avoid infinite loops
    state = {
        "financial_score": 0.50, 
        "financial_retry_count": 2, 
        "financial_context": ["Some SEC text"]
    }
    assert should_refine_financials(state) == "extract_risks"