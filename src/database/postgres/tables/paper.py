from sqlalchemy import Column, Date, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from ..base import Base


class Paper(Base):
    __tablename__ = "papers"

    paperId = Column(String, primary_key=True)
    doi = Column(String, nullable=True, index=True)
    title = Column(String)
    abstract = Column(Text)
    publication_date = Column(Date, nullable=True, index=True)
    cited_by_count = Column(Integer, nullable=True, index=True)
    referenced_works_count = Column(Integer, nullable=True)
    paper_type = Column(String, nullable=True)
    venue = Column(String, nullable=True, index=True)
    topics = Column(JSONB, nullable=True)
