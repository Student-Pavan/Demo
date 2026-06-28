from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import torch
from transformers import pipeline


@dataclass
class MCQItem:
    question: str
    correct_answer: str
    options: List[str]
    difficulty: Optional[str] = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _extract_qa_pairs(raw_output: str) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    # Try block-based extraction first
    blocks = re.split(r"\n\s*\n", raw_output.strip())
    for block in blocks:
        q_match = re.search(r"(?:Question|Q)\s*[\d.]*\s*:\s*(.+)", block, re.IGNORECASE)
        a_match = re.search(r"(?:Answer|A)\s*[\d.]*\s*:\s*(.+)", block, re.IGNORECASE)
        if not q_match or not a_match:
            continue
        question = _clean_text(q_match.group(1))
        answer = _clean_text(a_match.group(1))
        if not question or not answer:
            continue
        if "?" not in question:
            question = question.rstrip(".") + "?"
        pairs.append((question, answer))

    # Fallback: line-by-line scan
    if not pairs:
        lines = raw_output.strip().splitlines()
        last_q = None
        for line in lines:
            line = line.strip()
            q_match = re.match(r"(?:Q(?:uestion)?\s*[\d.]*\s*:)\s*(.+)", line, re.IGNORECASE)
            a_match = re.match(r"(?:A(?:nswer)?\s*[\d.]*\s*:)\s*(.+)", line, re.IGNORECASE)
            if q_match:
                last_q = _clean_text(q_match.group(1))
                if "?" not in last_q:
                    last_q = last_q.rstrip(".") + "?"
            elif a_match and last_q:
                pairs.append((last_q, _clean_text(a_match.group(1))))
                last_q = None

    return pairs


def _extract_questions(raw_output: str) -> List[str]:
    questions: List[str] = []
    seen = set()

    # Prefer explicit question lines first.
    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        q_match = re.match(r"(?:Q(?:uestion)?\s*[\d.)-]*\s*:?)\s*(.+)", line, re.IGNORECASE)
        candidate = q_match.group(1).strip() if q_match else line
        if "?" not in candidate:
            continue
        candidate = candidate[: candidate.rfind("?") + 1].strip()
        norm = _normalize(candidate)
        if norm not in seen:
            seen.add(norm)
            questions.append(candidate)

    # Fallback: pull any sentence ending with '?'.
    if not questions:
        for q in re.findall(r"([^?.!\n][^?\n]{4,}\?)", raw_output):
            candidate = _clean_text(q)
            norm = _normalize(candidate)
            if norm not in seen:
                seen.add(norm)
                questions.append(candidate)

    return questions


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    return [s.strip() for s in sentences if len(s.strip()) >= 35]


QUESTION_STYLE_TEMPLATES = [
    "Which statement best describes {topic}?",
    "How is {topic} characterized in the passage?",
    "What is the main idea about {topic}?",
    "Which interpretation of {topic} is most accurate?",
    "What can be inferred about {topic} from the passage?",
    "In context, what does the passage suggest about {topic}?",
]


def _topic_from_sentence(sentence: str) -> str:
    cleaned = sentence.strip().rstrip(". ")
    lead = re.split(r"[,;:()]", cleaned)[0].strip()
    words = lead.split()
    if not words:
        return "this concept"

    # Skip very common lead-in words to get a stronger topic phrase.
    stop = {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "in",
        "on",
        "at",
        "for",
        "to",
        "of",
        "and",
        "with",
        "by",
        "from",
    }
    filtered = [w for w in words if w.lower() not in stop]
    source = filtered if filtered else words
    return " ".join(source[: min(8, len(source))]).rstrip(",;:")


def _diversify_question(question: str, answer: str, index_seed: int) -> str:
    q = _clean_text(question)
    pattern = re.compile(r"^according to the passage,\s*what is true about\s*(.+)\?$", re.IGNORECASE)
    match = pattern.match(q)
    if not match:
        return q

    topic = _clean_text(match.group(1).strip()) or _topic_from_sentence(answer)
    template = QUESTION_STYLE_TEMPLATES[index_seed % len(QUESTION_STYLE_TEMPLATES)]
    return template.format(topic=topic)


def _heuristic_pairs_from_text(text: str, needed: int) -> List[tuple[str, str]]:
    """Create simple fallback QA pairs so MCQ generation can still proceed."""
    pairs: List[tuple[str, str]] = []
    for idx, sentence in enumerate(_split_sentences(text)):
        answer = sentence.rstrip(". ")
        topic = _topic_from_sentence(answer)
        if not topic:
            continue
        template = QUESTION_STYLE_TEMPLATES[idx % len(QUESTION_STYLE_TEMPLATES)]
        question = template.format(topic=topic)
        pairs.append((question, answer))
        if len(pairs) >= needed:
            break
    return pairs


def _fallback_distractors(correct_answer: str) -> List[str]:
    generic = [
        "All of the above",
        "None of the above",
        "Insufficient information provided",
        "A different concept from the passage",
        "An unrelated example",
        "A broader definition applies here",
    ]
    correct_norm = _normalize(correct_answer)
    distractors = [item for item in generic if _normalize(item) != correct_norm]
    return distractors[:3]


class QuestionGenerator:
    """Generate MCQs from source text with free HuggingFace transformer models."""

    # Free, HuggingFace-hosted models (no API key needed)
    DEFAULT_MODEL = "google/flan-t5-small"   # Lighter and faster first-run download
    ALT_MODEL = "google/flan-t5-base"        # Higher quality fallback

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_input_chars: int = 4000,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.max_input_chars = max_input_chars
        self.random = random.Random(seed)

    @property
    def generator(self):
        return _get_generator(self.model_name)

    def _generate_qa_pairs(self, text: str, max_questions: int) -> List[tuple[str, str]]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        clipped = cleaned[: self.max_input_chars]
        seen: set[tuple[str, str]] = set()
        unique: List[tuple[str, str]] = []

        def add_pair(question: str, answer: str):
            q = _clean_text(question)
            a = _clean_text(answer)
            if not q or not a:
                return
            if "?" not in q:
                q = q.rstrip(".") + "?"
            q = _diversify_question(q, a, len(unique))
            key = (_normalize(q), _normalize(a))
            if key not in seen:
                seen.add(key)
                unique.append((q, a))

        def recover_answers(questions: List[str]):
            for question in questions:
                answer_prompt = (
                    "Answer the question using only the passage. "
                    "Return only a short answer phrase.\n\n"
                    f"Passage: {clipped}\n"
                    f"Question: {question}\n"
                    "Answer:"
                )
                try:
                    ans_raw = self.generator(
                        answer_prompt,
                        max_new_tokens=36,
                        do_sample=False,
                        num_return_sequences=1,
                    )[0]["generated_text"]
                except Exception:
                    ans_raw = ""

                answer = _clean_text(ans_raw.splitlines()[0] if ans_raw else "")
                answer = re.sub(r"^(?:answer\s*:\s*)", "", answer, flags=re.IGNORECASE).strip()
                if answer:
                    add_pair(question, answer)
                if len(unique) >= max_questions:
                    return

        attempts = 4
        for attempt in range(attempts):
            remaining = max_questions - len(unique)
            if remaining <= 0:
                break

            request_count = max(remaining + 2, remaining)
            prompt = (
                f"Generate {request_count} quiz questions with answers from the passage. "
                "Use a mix of styles: definition, cause-effect, comparison, chronology, application, and inference. "
                "Avoid repeating the same opening phrase. Do not start every question with 'According to the passage'. "
                "Format each item exactly as:\nQuestion: <question text>\nAnswer: <short answer>\n\n"
                f"Passage: {clipped}"
            )

            try:
                result = self.generator(
                    prompt,
                    max_new_tokens=640,
                    do_sample=True,
                    temperature=0.7 + (attempt * 0.08),
                    top_p=0.92,
                    num_return_sequences=1,
                )
                raw = result[0]["generated_text"]
            except Exception:
                try:
                    raw = self.generator(prompt, max_new_tokens=320)[0]["generated_text"]
                except Exception:
                    raw = ""

            for q, a in _extract_qa_pairs(raw):
                add_pair(q, a)
                if len(unique) >= max_questions:
                    break

            if len(unique) >= max_questions:
                break

            questions_only = _extract_questions(raw)
            if questions_only:
                recover_answers(questions_only)

        if len(unique) < max_questions:
            for q, a in _heuristic_pairs_from_text(clipped, max_questions - len(unique)):
                add_pair(q, a)
                if len(unique) >= max_questions:
                    break

        return unique[:max_questions]

    def _build_options(self, correct_answer: str, answer_pool: List[str]) -> List[str]:
        correct_norm = _normalize(correct_answer)
        seen = {correct_norm}
        distractors: List[str] = []

        pool = list(answer_pool)
        self.random.shuffle(pool)
        for answer in pool:
            n = _normalize(answer)
            if n and n not in seen:
                seen.add(n)
                distractors.append(answer)
            if len(distractors) == 3:
                break

        if len(distractors) < 3:
            for fb in _fallback_distractors(correct_answer):
                if _normalize(fb) not in seen:
                    distractors.append(fb)
                    seen.add(_normalize(fb))
                if len(distractors) == 3:
                    break

        options = [correct_answer] + distractors[:3]
        self.random.shuffle(options)
        return options

    def generate_mcqs(
        self,
        text: str,
        max_questions: int = 5,
        difficulty_filter: Optional[str] = None,
        difficulty_classifier=None,
    ) -> List[MCQItem]:
        # Generate extra if filtering by difficulty
        fetch_count = max_questions * 3 if difficulty_filter and difficulty_classifier else max_questions
        qa_pairs = self._generate_qa_pairs(text, max_questions=fetch_count)
        answer_pool = [a for _, a in qa_pairs]

        mcqs: List[MCQItem] = []
        for question, correct_answer in qa_pairs:
            options = self._build_options(correct_answer, answer_pool)
            if len(options) < 4:
                continue

            diff = None
            if difficulty_classifier:
                try:
                    pred = difficulty_classifier.classify(question)
                    diff = pred.get("difficulty")
                except Exception:
                    pass

            # Filter by difficulty if requested
            if difficulty_filter and diff:
                if diff.lower() != difficulty_filter.lower():
                    continue

            mcqs.append(
                MCQItem(
                    question=question,
                    correct_answer=correct_answer,
                    options=options,
                    difficulty=diff,
                )
            )

            if len(mcqs) >= max_questions:
                break

        return mcqs[:max_questions]


@lru_cache(maxsize=2)
def _get_generator(model_name: str):
    device = 0 if torch.cuda.is_available() else -1
    last_error = None
    for task_name in ("text2text-generation", "any-to-any"):
        try:
            return pipeline(
                task_name,
                model=model_name,
                device=device,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not initialize generation pipeline for model '{model_name}'."
    ) from last_error
