from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils import config
from .base import Base

_engine = None
_sessionmaker = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(config.DB_URL, echo=False)
    return _engine

def get_session():
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = sessionmaker(bind=get_engine())
    return _sessionmaker()

def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())
