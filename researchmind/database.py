import os
import sqlite3

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.abspath(os.path.join(DB_DIR, "research_history.db"))

os.makedirs(DB_DIR, exist_ok=True)

conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS reports(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT,
    report TEXT,
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()


def save_report(
    topic,
    report,
    feedback
):

    cursor.execute(
        """
        INSERT INTO reports
        (
            topic,
            report,
            feedback
        )
        VALUES (?, ?, ?)
        """,
        (
            topic,
            report,
            feedback
        )
    )

    conn.commit()


def get_reports():

    cursor.execute("""
        SELECT *
        FROM reports
        ORDER BY created_at DESC
    """)

    return cursor.fetchall()