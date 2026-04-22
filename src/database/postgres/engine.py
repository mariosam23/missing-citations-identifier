from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils import config
from .base import Base

engine = create_engine(config.DB_URL, echo=False)
Session = sessionmaker(bind=engine)
session = Session()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
