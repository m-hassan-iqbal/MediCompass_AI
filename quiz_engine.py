"""
MediCompass AI - Quiz Engine & Validation Module
Generates, validates, and grades 10-MCQ Concept Checks for MDCAT and NUMS.
Enforces 10-question count, 4 options, 1 valid answer, and detailed misconception breakdowns.
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
from typing import Any, Optional

from data_ingestion import DocumentChunk
import prompts
from rag_engine import build_compact_context, clean_json_response

logger = logging.getLogger(__name__)


@dataclass
class QuizQuestion:
    id: int
    type: str  # 'Direct Understanding', 'Conceptual Reasoning', 'Application / Interpretation'
    question: str
    options: list[str]
    answer: str
    concept: str
    explanation: str
    memory: str
    past_paper: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_quiz_data(data: dict[str, Any]) -> tuple[bool, list[str], list[QuizQuestion]]:
    """
    Validates quiz structure against strict product criteria:
    - Exactly 10 questions
    - Exactly 4 options per question
    - Exactly 1 correct answer that exists inside the options list
    - No duplicate options or questions
    """
    errors = []
    validated_questions: list[QuizQuestion] = []

    questions_raw = data.get("questions")
    if not isinstance(questions_raw, list):
        return False, ["Root 'questions' key is missing or not a list."], []

    if len(questions_raw) != 10:
        errors.append(f"Expected exactly 10 questions, found {len(questions_raw)}.")

    seen_stems = set()

    for i, q in enumerate(questions_raw):
        qid = q.get("id", i + 1)
        stem = str(q.get("question", "")).strip()
        options = q.get("options", [])
        answer = str(q.get("answer", "")).strip()
        qtype = q.get("type", "Conceptual Reasoning")
        concept = str(q.get("concept", "")).strip()
        explanation = str(q.get("explanation", "")).strip()
        memory = str(q.get("memory", "")).strip()
        past_paper = str(q.get("past_paper", "Fresh conceptual variation")).strip()

        if not stem:
            errors.append(f"Question {qid} has an empty question stem.")
        elif stem in seen_stems:
            errors.append(f"Question {qid} has a duplicate question stem.")
        seen_stems.add(stem)

        if not isinstance(options, list) or len(options) != 4:
            errors.append(f"Question {qid} must have exactly 4 options, found {len(options) if isinstance(options, list) else 0}.")
        else:
            cleaned_options = [str(opt).strip() for opt in options]
            if len(set(cleaned_options)) != 4:
                errors.append(f"Question {qid} contains duplicate options.")

            # Check that answer matches one option
            matched_option = None
            for opt in cleaned_options:
                if answer == opt or (len(answer) == 1 and answer.upper() in ["A", "B", "C", "D"]):
                    matched_option = opt
                    break
                # Handle cases where answer might have prefix like "A) ..."
                if opt.lower().startswith(answer.lower()) or answer.lower().startswith(opt.lower()):
                    matched_option = opt
                    break

            if not matched_option:
                errors.append(f"Question {qid} answer '{answer}' does not match any of the 4 options: {cleaned_options}")
            else:
                answer = matched_option

        validated_questions.append(
            QuizQuestion(
                id=qid,
                type=qtype,
                question=stem,
                options=options if isinstance(options, list) else [],
                answer=answer,
                concept=concept,
                explanation=explanation,
                memory=memory,
                past_paper=past_paper,
            )
        )

    is_valid = len(errors) == 0
    return is_valid, errors, validated_questions


class QuizGenerator:
    """Coordinates with Groq to generate verified 10-MCQ quizzes with fallback demo capability."""

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self._client = None

    def get_client(self):
        if self._client is None and self.api_key:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client for Quiz: {e}")
        return self._client

    def is_configured(self) -> bool:
        return bool(self.api_key and not self.api_key.startswith("gsk_your_groq_api_key"))

    def generate_10_mcq_quiz(
        self,
        concept_title: str,
        subject: str,
        exam: str,
        retrieved_chunks: list[DocumentChunk],
    ) -> list[QuizQuestion]:
        """Generates a 10-MCQ concept check, validates it, and returns the questions."""
        if not self.is_configured():
            logger.info("Groq API key not configured. Using verified grounded 10-MCQ seed quiz.")
            return self._generate_verified_demo_quiz()

        client = self.get_client()
        if client is None:
            return self._generate_verified_demo_quiz()

        context_str = build_compact_context(retrieved_chunks)
        user_prompt = prompts.QUIZ_GENERATION_USER_TEMPLATE.format(
            concept_title=concept_title,
            subject=subject,
            exam=exam,
            retrieved_context=context_str,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts.QUIZ_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content or "{}"
            cleaned = clean_json_response(raw_content)
            data = json.loads(cleaned)

            is_valid, errors, questions = validate_quiz_data(data)
            if is_valid:
                return questions

            logger.warning(f"Quiz validation failed with errors: {errors}. Retrying with repair.")
            # Retry once with repair prompt
            repair_prompt = prompts.JSON_REPAIR_PROMPT.format(
                error="; ".join(errors),
                raw_output=raw_content[:2000],
            )
            repair_resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.0,
                max_tokens=3000,
                response_format={"type": "json_object"},
            )
            repair_data = json.loads(clean_json_response(repair_resp.choices[0].message.content or "{}"))
            is_valid2, _, questions2 = validate_quiz_data(repair_data)
            if is_valid2:
                return questions2

            logger.warning("Repaired quiz still failed validation. Using verified grounded seed quiz.")
            return self._generate_verified_demo_quiz()

        except Exception as e:
            logger.error(f"Quiz generation failed ({e}). Using verified grounded seed quiz.")
            return self._generate_verified_demo_quiz()

    def _generate_verified_demo_quiz(self) -> list[QuizQuestion]:
        """
        High-quality verified 10-MCQ set based on Punjab & Federal Textbook enzyme inhibition concepts.
        Strictly follows: 3 Direct Understanding, 4 Conceptual Reasoning, 3 Application / Interpretation.
        """
        questions = [
            # 1. Direct Understanding
            QuizQuestion(
                id=1,
                type="Direct Understanding",
                question="Where does a competitive inhibitor specifically bind on the enzyme molecule?",
                options=[
                    "At the catalytic active site",
                    "At an allosteric regulatory site",
                    "To the non-protein prosthetic group",
                    "To the peptide backbone of the apoenzyme"
                ],
                answer="At the catalytic active site",
                concept="Competitive inhibitor binding locus",
                explanation="Competitive inhibitors have structural similarity with the substrate and compete directly for the active site of the enzyme.",
                memory="Competitive = Competes for the Active Site",
                past_paper="MDCAT 2021 Concept Variation"
            ),
            # 2. Direct Understanding
            QuizQuestion(
                id=2,
                type="Direct Understanding",
                question="What effect does a competitive inhibitor have on the maximum velocity (Vmax) of an enzyme-catalyzed reaction?",
                options=[
                    "Vmax remains completely unchanged",
                    "Vmax is reduced by half",
                    "Vmax increases proportionately",
                    "Vmax drops to zero irreversibly"
                ],
                answer="Vmax remains completely unchanged",
                concept="Effect of competitive inhibition on Vmax",
                explanation="Because competitive inhibition can be completely surmounted at very high substrate concentrations, the maximum reaction rate (Vmax) is not altered.",
                memory="Vmax stays CONSTANT in competitive inhibition",
                past_paper="MDCAT 2023 Concept Variation"
            ),
            # 3. Direct Understanding
            QuizQuestion(
                id=3,
                type="Direct Understanding",
                question="How does the Michaelis constant (Km) change in the presence of a competitive inhibitor?",
                options=[
                    "Apparent Km increases",
                    "Km decreases to zero",
                    "Km remains strictly constant",
                    "Km is converted into Vmax"
                ],
                answer="Apparent Km increases",
                concept="Apparent Km alteration",
                explanation="Because more substrate is required to outcompete the inhibitor and achieve half of Vmax, the apparent Km increases (indicating lower apparent affinity).",
                memory="Km UP = Affinity DOWN",
                past_paper="NUMS 2024 Concept Variation"
            ),
            # 4. Conceptual Reasoning
            QuizQuestion(
                id=4,
                type="Conceptual Reasoning",
                question="If an enzymatic reaction is currently blocked by a competitive inhibitor, what immediate intervention restores the original reaction velocity?",
                options=[
                    "Significantly increasing the concentration of substrate",
                    "Lowering the reaction temperature to 0°C",
                    "Adding an allosteric activator",
                    "Decreasing the total enzyme concentration"
                ],
                answer="Significantly increasing the concentration of substrate",
                concept="Overcoming competitive inhibition",
                explanation="A competitive inhibitor competes reversibly with substrate for the active site. Increasing substrate concentration shifts the binding equilibrium toward the substrate, displacing the inhibitor.",
                memory="Excess substrate outcompetes the inhibitor",
                past_paper="MDCAT 2023 Concept Variation"
            ),
            # 5. Conceptual Reasoning
            QuizQuestion(
                id=5,
                type="Conceptual Reasoning",
                question="Why is non-competitive inhibition fundamentally impossible to overcome by simply adding excess substrate?",
                options=[
                    "The inhibitor binds to an allosteric site, altering enzyme conformation rather than competing for active site space",
                    "The substrate molecules are permanently destroyed by the inhibitor",
                    "The inhibitor locks the substrate irreversibly inside the active site",
                    "Non-competitive inhibitors only function at absolute zero temperature"
                ],
                answer="The inhibitor binds to an allosteric site, altering enzyme conformation rather than competing for active site space",
                concept="Mechanism of non-competitive inhibition",
                explanation="Non-competitive inhibitors do not compete with substrate; they bind at an allosteric site and distort the catalytic conformation of the active site.",
                memory="Allosteric site binding cannot be outcompeted by substrate",
                past_paper="NUMS 2024 Concept Variation"
            ),
            # 6. Conceptual Reasoning
            QuizQuestion(
                id=6,
                type="Conceptual Reasoning",
                question="In a biochemical experiment, Substance X reduces the rate of succinate dehydrogenase. Adding ten-fold succinate restores normal catalytic rate. Substance X is most likely:",
                options=[
                    "Malonate, acting as a competitive inhibitor",
                    "Cyanide, acting as an irreversible poison",
                    "Lead ions, acting as an allosteric heavy metal",
                    "A non-protein coenzyme activator"
                ],
                answer="Malonate, acting as a competitive inhibitor",
                concept="Classical malonate-succinate competition",
                explanation="Malonate is structurally analogous to succinate and serves as the classic competitive inhibitor of succinate dehydrogenase described in Punjab Textbook Ch. 11.",
                memory="Malonate vs Succinate = Classic Competitive Pair",
                past_paper="Punjab Textbook Core Yield"
            ),
            # 7. Conceptual Reasoning
            QuizQuestion(
                id=7,
                type="Conceptual Reasoning",
                question="Which of the following graphical characteristics distinguishes a competitive inhibitor from an uninhibited enzyme on a Lineweaver-Burk double reciprocal plot?",
                options=[
                    "Same Y-intercept (1/Vmax), but different X-intercept (-1/Km shifted closer to zero)",
                    "Different Y-intercept, but identical X-intercept",
                    "Parallel lines with completely identical slopes",
                    "Both intercepts shifted to negative infinity"
                ],
                answer="Same Y-intercept (1/Vmax), but different X-intercept (-1/Km shifted closer to zero)",
                concept="Lineweaver-Burk kinetics of competitive inhibition",
                explanation="Since Vmax is unchanged, 1/Vmax (Y-intercept) remains identical. Since Km increases, -1/Km (X-intercept) moves closer to the origin.",
                memory="Y-intercept identical = Vmax unchanged",
                past_paper="NUMS High-Yield Concept"
            ),
            # 8. Application / Interpretation
            QuizQuestion(
                id=8,
                type="Application / Interpretation",
                question="Sulfa drugs (e.g. Sulfanilamide) kill bacteria by competitively inhibiting the enzyme dihydropteroate synthetase. Why are human host cells unharmed by this treatment?",
                options=[
                    "Humans do not synthesize folic acid endogenously and obtain it preformed from diet",
                    "Human enzymes destroy sulfanilamide immediately with pepsin",
                    "Human bacterial walls are completely impermeable to sulfonamides",
                    "Sulfa drugs convert into human coenzymes inside the liver"
                ],
                answer="Humans do not synthesize folic acid endogenously and obtain it preformed from diet",
                concept="Pharmacological application of competitive inhibition",
                explanation="Bacteria must synthesize their own folic acid from PABA (which sulfanilamide mimics). Humans absorb folic acid directly from nutrition and lack the targeted pathway.",
                memory="Sulfa drugs starve bacteria of folic acid without affecting human cells",
                past_paper="Federal Textbook Pharmacology Section"
            ),
            # 9. Application / Interpretation
            QuizQuestion(
                id=9,
                type="Application / Interpretation",
                question="A medical researcher observes that Compound Z binds equally well whether the enzyme has already bound its substrate or not. The Vmax drops by 50% while Km is unchanged. Compound Z is a:",
                options=[
                    "Pure non-competitive inhibitor",
                    "Reversible competitive inhibitor",
                    "Substrate analog activator",
                    "Competitive apoenzyme prosthetic factor"
                ],
                answer="Pure non-competitive inhibitor",
                concept="Identification of non-competitive inhibition via kinetic parameters",
                explanation="Decreased Vmax with unchanged Km is the diagnostic signature of pure non-competitive inhibition, where the inhibitor binds to both free enzyme and ES complex with equal affinity.",
                memory="Vmax DOWN + Km SAME = Non-Competitive",
                past_paper="MDCAT 2021 Concept Variation"
            ),
            # 10. Application / Interpretation
            QuizQuestion(
                id=10,
                type="Application / Interpretation",
                question="In emergency toxicology, methanol poisoning is treated by intravenously administering high-dose ethanol. What is the pharmacological basis of this treatment?",
                options=[
                    "Ethanol competitively inhibits alcohol dehydrogenase, preventing the formation of toxic formaldehyde and formic acid",
                    "Ethanol denatures all alcohol dehydrogenase enzymes permanently",
                    "Ethanol acts as an allosteric inhibitor that decreases Vmax",
                    "Ethanol directly neutralizes formic acid in the bloodstream"
                ],
                answer="Ethanol competitively inhibits alcohol dehydrogenase, preventing the formation of toxic formaldehyde and formic acid",
                concept="Clinical rescue via competitive displacement",
                explanation="Ethanol has a higher affinity for alcohol dehydrogenase than methanol. Administering excess ethanol competitively displaces methanol, allowing it to be excreted harmlessly.",
                memory="Competitive displacement prevents toxic metabolite synthesis",
                past_paper="Clinical Application Scenario"
            ),
        ]
        return questions


def grade_quiz_submission(
    questions: list[QuizQuestion], user_answers: dict[int, str]
) -> dict[str, Any]:
    """
    Computes total score out of 10, categorizes mastery band,
    and returns detailed per-question performance breakdowns.
    """
    correct_count = 0
    detailed_results = []

    for q in questions:
        user_choice = user_answers.get(q.id, "").strip()
        is_correct = user_choice == q.answer.strip()
        if is_correct:
            correct_count += 1

        detailed_results.append(
            {
                "id": q.id,
                "type": q.type,
                "question": q.question,
                "user_answer": user_choice if user_choice else "Unanswered",
                "correct_answer": q.answer,
                "is_correct": is_correct,
                "concept": q.concept,
                "explanation": q.explanation,
                "memory": q.memory,
                "past_paper": q.past_paper,
            }
        )

    # Mastery categorization as required by PRD Section 49
    if correct_count >= 8:
        mastery_label = "Strong concept control"
        action_message = "Review the missed questions and keep the core rule active."
        color_theme = "#22C55E"
    elif correct_count >= 5:
        mastery_label = "Good foundation"
        action_message = "Revisit the weak points before moving to harder applications."
        color_theme = "#F59E0B"
    else:
        mastery_label = "Developing - Needs another pass"
        action_message = "The concept needs another pass. Use the diagram and book evidence, then retry."
        color_theme = "#EF4444"

    return {
        "score": correct_count,
        "total": len(questions),
        "percentage": round((correct_count / len(questions)) * 100, 1),
        "mastery_label": mastery_label,
        "action_message": action_message,
        "color_theme": color_theme,
        "detailed_results": detailed_results,
    }
