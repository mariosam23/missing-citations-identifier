from database import engine, Base, SessionLocal

def test_db_connection():
    try:
        with engine.connect() as connection:
            print("Database connection successful!")
    except Exception as e:
        print("Database connection failed:", str(e))

if __name__ == "__main__":
    test_db_connection()