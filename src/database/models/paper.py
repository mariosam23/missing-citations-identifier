from sqlalchemy import Column, Integer, String, Text, Date
from sqlalchemy.dialects.postgresql import JSONB
from ..database import Base

class Paper(Base):
    __tablename__ = 'papers'

    paperId = Column(String, primary_key=True)
    doi = Column(String, nullable=True, index=True)
    title = Column(String)
    abstract = Column(Text)
    publication_date = Column(Date, nullable=True, index=True)
    cited_by_count = Column(Integer, nullable=True, index=True)
    paper_type = Column(String, nullable=True)
    concepts = Column(JSONB, nullable=True)
