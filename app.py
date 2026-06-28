from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import threading
import uuid
import urllib.parse
import webbrowser
from pathlib import Path

from database import get_recent_quiz_sessions, init_db, insert_quiz_session
from difficulty_classifier import DifficultyClassifier
from question_generator import MCQItem, QuestionGenerator

UI_FILE = Path(__file__).with_name("adaptiq-premium_fixed (7).html")
UI_ROUTE = "AI_ADAPTIVEQUIZ"
DEFAULT_PORT = int(os.getenv("PORT", "7860"))

question_generator = QuestionGenerator()
init_db()
try:
    difficulty_classifier = DifficultyClassifier()
except FileNotFoundError:
    difficulty_classifier = None


def _estimate_difficulty(question: str) -> str:
    text = question.strip()
    if difficulty_classifier:
        try:
            pred = difficulty_classifier.classify(text)
            return str(pred.get("difficulty", "medium")).strip().lower()
        except Exception:
            pass

    words = len(text.split())
    if words <= 9:
        return "easy"
    if words <= 15:
        return "medium"
    return "hard"


def _title_diff(diff: str) -> str:
    d = (diff or "medium").strip().lower()
    return {"easy": "Easy", "medium": "Medium", "hard": "Hard"}.get(d, "Medium")


def _bloom_for(diff: str) -> str:
    d = (diff or "medium").lower()
    if d == "easy":
        return "Understand (L2)"
    if d == "hard":
        return "Analyze (L4)"
    return "Apply (L3)"


def _to_frontend_item(item: MCQItem) -> dict:
    difficulty = _title_diff(item.difficulty or _estimate_difficulty(item.question))
    options = list(item.options or [])
    if not options:
        options = [item.correct_answer]
    try:
        correct_idx = options.index(item.correct_answer)
    except ValueError:
        options = [item.correct_answer] + options[:3]
        correct_idx = 0

    return {
        "id": str(uuid.uuid4()),
        "question": item.question,
        "type": "MCQ",
        "difficulty": difficulty,
        "bloom": _bloom_for(difficulty),
        "options": options,
        "correct": correct_idx,
        "answer": item.correct_answer,
        "keyword": item.correct_answer.split(" ")[0] if item.correct_answer else "concept",
        "sentence": item.correct_answer,
    }


class AppHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        normalized = parsed.path.rstrip("/")

        if parsed.path == "/api/sessions":
            self._handle_list_sessions(parsed.query)
            return

        route_aliases = {
            f"/{UI_ROUTE}",
            f"/{UI_ROUTE}.html",
        }

        if normalized in route_aliases:
            self.path = f"/{UI_FILE.name}"
            return super().do_GET()

        if parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", f"/{UI_ROUTE}")
            self.end_headers()
            return

        return super().do_GET()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path == "/api/generate":
            self._handle_generate()
            return
        if self.path == "/api/session":
            self._handle_save_session()
            return
        self._send_json({"error": "Not found"}, status=404)

    def _handle_list_sessions(self, query: str) -> None:
        try:
            query_data = urllib.parse.parse_qs(query)
            limit = int(query_data.get("limit", ["20"])[0])
            limit = max(1, min(limit, 100))

            rows = get_recent_quiz_sessions(limit=limit)
            sessions = [
                {
                    "id": int(row["id"]),
                    "total_questions": int(row["total_questions"]),
                    "correct_answers": int(row["correct_answers"]),
                    "percentage": float(row["percentage"]),
                    "difficulty_filter": row["difficulty_filter"] or "All",
                    "input_type": row["input_type"] or "Unknown",
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
            self._send_json({"sessions": sessions, "count": len(sessions)})
        except Exception as exc:
            self._send_json({"error": f"Failed to list sessions: {exc}"}, status=500)

    def _handle_save_session(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))

            total_questions = int(payload.get("total_questions", 0))
            correct_answers = int(payload.get("correct_answers", 0))
            percentage = float(payload.get("percentage", 0.0))
            difficulty_filter = str(payload.get("difficulty_filter", "Adaptive"))
            input_type = str(payload.get("input_type", "Text"))

            if total_questions <= 0:
                self._send_json({"error": "total_questions must be greater than 0"}, status=400)
                return
            if correct_answers < 0 or correct_answers > total_questions:
                self._send_json({"error": "correct_answers out of range"}, status=400)
                return

            session_id = insert_quiz_session(
                total_questions=total_questions,
                correct_answers=correct_answers,
                percentage=percentage,
                difficulty_filter=difficulty_filter,
                input_type=input_type,
            )
            self._send_json({"ok": True, "session_id": session_id})
        except Exception as exc:
            self._send_json({"error": f"Failed to save session: {exc}"}, status=500)

    def _handle_generate(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))

            text = str(payload.get("text", "")).strip()
            if len(text) < 80:
                self._send_json(
                    {"error": "Please provide at least 80 characters of study content."},
                    status=400,
                )
                return

            count = int(payload.get("count", 10))
            count = max(1, min(20, count))

            difficulty_mode = str(payload.get("difficulty", "Adaptive")).strip()
            difficulty_filter = None if difficulty_mode.lower() == "adaptive" else difficulty_mode

            generated = question_generator.generate_mcqs(
                text,
                max_questions=count,
                difficulty_filter=difficulty_filter,
                difficulty_classifier=difficulty_classifier,
            )

            if not generated:
                self._send_json(
                    {"error": "Could not generate questions from this content. Try richer text."},
                    status=500,
                )
                return

            questions = [_to_frontend_item(item) for item in generated]
            self._send_json(
                {
                    "questions": questions,
                    "meta": {
                        "count": len(questions),
                        "difficulty": difficulty_mode,
                        "source": "python-nlp-backend",
                    },
                }
            )
        except Exception as exc:
            self._send_json({"error": f"Generation failed: {exc}"}, status=500)


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the exact AdaptIQ UI HTML file.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run the UI server on.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser.",
    )
    return parser.parse_args()


def serve(port: int, open_browser: bool = True) -> None:
    if not UI_FILE.exists():
        raise FileNotFoundError(
            f"Required UI file was not found: {UI_FILE}. "
            "Place the provided HTML file in the project root."
        )

    # Serve files from project root so CSS/JS in the HTML keep working exactly.
    root_dir = UI_FILE.parent
    handler = AppHandler

    os.chdir(root_dir)
    with ReusableTCPServer(("", port), handler) as httpd:
        url = f"http://0.0.0.0:{port}/{UI_ROUTE}"
        legacy_url = f"http://127.0.0.1:{port}/{UI_FILE.name}"
        print(f"Serving UI at: {url}")
        print(f"Legacy URL still works: {legacy_url}")

        if open_browser and not os.environ.get("SPACE_ID"):
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    args = parse_args()
    serve(
        port=int(os.getenv("PORT", args.port)),
        open_browser=False
    )
