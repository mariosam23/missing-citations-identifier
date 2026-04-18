from sqlalchemy import Column, ForeignKey, String

from base import Base


class Citation(Base):
    __tablename__ = "citations"

    source_paper_id = Column(String, ForeignKey("papers.paperId"), primary_key=True)
    target_paper_id = Column(String, primary_key=True, index=True)
