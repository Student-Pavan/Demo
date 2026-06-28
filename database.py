from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import List, Optional

DB_PATH = Path(os.getenv("SQLITE_DB_PATH", "quiz_data.db"))


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_questions INTEGER NOT NULL,
                correct_answers INTEGER NOT NULL,
                percentage REAL NOT NULL,
                difficulty_filter TEXT,
                input_type TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS question_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                student_response TEXT,
                time_taken REAL,
                accuracy REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def insert_quiz_session(
    total_questions: int,
    correct_answers: int,
    percentage: float,
    difficulty_filter: Optional[str],
    input_type: Optional[str],
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO quiz_sessions (
                total_questions,
                correct_answers,
                percentage,
                difficulty_filter,
                input_type
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (total_questions, correct_answers, percentage, difficulty_filter, input_type),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_recent_quiz_sessions(limit: int = 10) -> List[sqlite3.Row]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                total_questions,
                correct_answers,
                percentage,
                difficulty_filter,
                input_type,
                created_at
            FROM quiz_sessions
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return rows


def insert_generated_question(question: str, difficulty: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO question_attempts (question, difficulty)
            VALUES (?, ?)
            """,
            (question, difficulty),
        )
        conn.commit()
        return int(cursor.lastrowid)


def update_student_response(
    attempt_id: int,
    student_response: str,
    time_taken: Optional[float],
    accuracy: Optional[float],
) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE question_attempts
            SET student_response = ?,
                time_taken = ?,
                accuracy = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (student_response, time_taken, accuracy, attempt_id),
        )
        conn.commit()
        return cursor.rowcount > 0
