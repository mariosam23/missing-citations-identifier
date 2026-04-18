from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.sql import func

from base import Base


class CorpusPaper(Base):
    __tablename__ = "corpus_papers"

    paper_id = Column(String, ForeignKey("papers.paperId"), primary_key=True)
    role = Column(String, nullable=False, index=True) # seed, hop1, hop2
    in_eval_set = Column(Boolean, nullable=False, default=False)
    reference_coverage = Column(Float, nullable=True)
    indexed_reference_count = Column(Integer, nullable=True)
    added_at = Column(DateTime, nullable=False, default=func.now())
