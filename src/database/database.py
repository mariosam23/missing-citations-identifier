from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import config

engine = create_engine(config.DB_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
