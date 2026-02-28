from fetch_papers import fetch_and_store_papers
from database import init_db


def main():
    init_db()
    fetch_and_store_papers()


if __name__ == "__main__":
    main()
