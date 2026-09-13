"""
Unit tests for 10-MCQ Quiz validation and scoring engine.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from quiz_engine import validate_quiz_data, grade_quiz_submission, QuizQuestion


def create_mock_valid_quiz_data(num_questions=10, num_options=4):
    questions = []
    for i in range(1, num_questions + 1):
        opts = [f"Option {chr(65 + j)} for Q{i}" for j in range(num_options)]
        questions.append({
            "id": i,
            "type": "Conceptual Reasoning",
            "question": f"Question stem for item {i}?",
            "options": opts,
            "answer": opts[0],
            "concept": f"Concept {i}",
            "explanation": f"Explanation {i}",
            "memory": f"Memory {i}",
            "past_paper": "MDCAT 2023",
        })
    return {"questions": questions}


def test_valid_quiz_passes():
    data = create_mock_valid_quiz_data(10, 4)
    is_valid, errors, validated = validate_quiz_data(data)
    assert is_valid is True
    assert len(errors) == 0
    assert len(validated) == 10


def test_reject_less_than_10_questions():
    data = create_mock_valid_quiz_data(8, 4)
    is_valid, errors, _ = validate_quiz_data(data)
    assert is_valid is False
    assert any("Expected exactly 10 questions" in err for err in errors)


def test_reject_invalid_option_count():
    data = create_mock_valid_quiz_data(10, 3)
    is_valid, errors, _ = validate_quiz_data(data)
    assert is_valid is False
    assert any("must have exactly 4 options" in err for err in errors)


def test_reject_answer_not_in_options():
    data = create_mock_valid_quiz_data(10, 4)
    data["questions"][0]["answer"] = "Completely different alien option"
    is_valid, errors, _ = validate_quiz_data(data)
    assert is_valid is False
    assert any("does not match any of the 4 options" in err for err in errors)


def test_reject_duplicate_options():
    data = create_mock_valid_quiz_data(10, 4)
    data["questions"][1]["options"][1] = data["questions"][1]["options"][0]
    is_valid, errors, _ = validate_quiz_data(data)
    assert is_valid is False
    assert any("contains duplicate options" in err for err in errors)


def test_grading_scoring_and_mastery_bands():
    data = create_mock_valid_quiz_data(10, 4)
    _, _, questions = validate_quiz_data(data)

    answers_perfect = {q.id: q.answer for q in questions}
    result_perfect = grade_quiz_submission(questions, answers_perfect)
    assert result_perfect["score"] == 10
    assert result_perfect["percentage"] == 100.0
    assert "Strong concept control" in result_perfect["mastery_label"]

    answers_medium = {q.id: q.answer for q in questions[:6]}
    result_medium = grade_quiz_submission(questions, answers_medium)
    assert result_medium["score"] == 6
    assert result_medium["percentage"] == 60.0
    assert "Good foundation" in result_medium["mastery_label"]

    answers_low = {q.id: q.answer for q in questions[:3]}
    result_low = grade_quiz_submission(questions, answers_low)
    assert result_low["score"] == 3
    assert result_low["percentage"] == 30.0
    assert "Needs another pass" in result_low["mastery_label"]
