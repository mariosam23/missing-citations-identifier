from sqlalchemy import Column, String, Float, Integer, ForeignKey
from sqlalchemy.orm import relationship
from src.database.models import Base

class Concept(Base):
    __tablename__ = 'concepts'
    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(String, ForeignKey('papers.id'), index=True)
    label = Column(String, index=True)
    concept_type = Column(String)
    salience_score = Column(Float)
