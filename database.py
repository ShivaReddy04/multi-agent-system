import sqlite3

conn = sqlite3.connect(
    "research_history.db",
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