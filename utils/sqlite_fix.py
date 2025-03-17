"""SQLite version fix for ChromaDB"""
import sys

def fix_sqlite():
    """Fix SQLite version issues by using pysqlite3 if available"""
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
    except ImportError:
        pass 