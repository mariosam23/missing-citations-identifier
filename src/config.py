import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DB_URL = os.getenv('DB_URL', "")
    OPEN_ALEX_API_KEY = os.getenv('OPEN_ALEX_API_KEY', "")
    OPEN_ALEX_EMAIL = os.getenv('OPEN_ALEX_EMAIL', "")

config = Config()