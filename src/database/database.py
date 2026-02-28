from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import config

engine = create_engine(config.DB_URL, echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()

def init_db():
    import database.models.paper
    import database.models.citation
    Base.metadata.create_all(bind=engine)

