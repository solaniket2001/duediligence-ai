import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from src.db.database import Base

class DueDiligenceReport(Base):
    __tablename__ = "due_diligence_reports"

    id = Column(Integer, primary_key=True, index=True)
    target_ticker = Column(String, index=True, nullable=False)
    target_name = Column(String, nullable=False)
    peer_ticker = Column(String, index=True, nullable=True)
    
    financial_score = Column(Float, nullable=True)
    risk_score = Column(Integer, nullable=True)
    
    quantitative_analysis = Column(Text, nullable=True)
    news_context = Column(Text, nullable=True)
    final_memo = Column(Text, nullable=False)
    
    market_data = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)