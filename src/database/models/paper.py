from sqlalchemy import Column, String, Text
from ..database import Base

class Paper(Base):
    __tablename__ = 'papers'

    paperId = Column(String, primary_key=True)
    title = Column(String)
    abstract = Column(Text)
