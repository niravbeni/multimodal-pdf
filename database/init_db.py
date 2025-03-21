from models import Base
from config import engine

def init_database():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("Initializing database...")
    init_database()
    print("Database initialized successfully!") 