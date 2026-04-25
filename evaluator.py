import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    id: str
    document_text: str
    question: str
    expected_keywords: list
    expected_source_cited: bool
    description: str


@dataclass
class TestResult:
    case_id: str
    description: str
    passed: bool
    answer: str
    missing_keywords: list
    citation_check_passed: bool
    error: Optional[str] = None


@dataclass
class EvalReport:
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list


BUILT_IN_TEST_CASES = [
    TestCase(
        id="TC01",
        document_text=(
            "The Eiffel Tower is located in Paris, France. "
            "It was built in 1889 by Gustave Eiffel for the World's Fair. "
            "The tower is 330 meters tall and receives about 7 million visitors per year."
        ),
        question="Who built the Eiffel Tower and when?",
        expected_keywords=["Eiffel", "1889"],
        expected_source_cited=True,
        description="Factual retrieval: named entity + date",
    ),
    TestCase(
        id="TC02",
        document_text=(
            "Python is a high-level programming language created by Guido van Rossum "
            "and first released in 1991. It emphasizes code readability and simplicity. "
            "Python supports multiple programming paradigms including procedural, "
            "object-oriented, and functional programming."
        ),
        question="What programming paradigms does Python support?",
        expected_keywords=["procedural", "object-oriented", "functional"],
        expected_source_cited=True,
        description="Multi-term retrieval: list enumeration",
    ),
    TestCase(
        id="TC03",
        document_text=(
            "Photosynthesis is the process by which plants convert sunlight into glucose. "
            "It occurs in the chloroplasts of plant cells. "
            "The overall equation is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2."
        ),
        question="Where does photosynthesis occur in plant cells?",
        expected_keywords=["chloroplast"],
        expected_source_cited=True,
        description="Single-term location retrieval",
    ),
    TestCase(
        id="TC04",
        document_text=(
            "The mitochondria is the powerhouse of the cell. "
            "It produces ATP through a process called cellular respiration. "
            "Humans have approximately 37 trillion cells in their bodies."
        ),
        question="What does the mitochondria produce?",
        expected_keywords=["ATP"],
        expected_source_cited=True,
        description="Short-doc term retrieval",
    ),
    TestCase(
        id="TC05",
        document_text=(
            "Machine learning is a subset of artificial intelligence. "
            "Supervised learning uses labeled training data. "
            "Unsupervised learning finds patterns without labels. "
            "Reinforcement learning trains agents via reward signals."
        ),
        question="How does reinforcement learning train its agents?",
        expected_keywords=["reward"],
        expected_source_cited=True,
        description="Targeted clause retrieval within multi-topic doc",
    ),
]


class EvalSuite:

    def __init__(self, rag_engine):
        self.engine = rag_engine
        self.test_cases = BUILT_IN_TEST_CASES

    def run_single(self, case: TestCase) -> TestResult:
        try:
            self.engine.build_index(case.document_text)
            response = self.engine.query(case.question)
            answer = response.answer

            answer_lower = answer.lower()
            missing_keywords = [
                kw for kw in case.expected_keywords if kw.lower() not in answer_lower
            ]
            keyword_check = len(missing_keywords) == 0

            citation_check = True
            if case.expected_source_cited:
                citation_check = "[source" in answer_lower

            passed = keyword_check and citation_check

            logger.info(
                "TC %s (%s): %s | missing=%s | cited=%s",
                case.id,
                case.description,
                "PASS" if passed else "FAIL",
                missing_keywords,
                citation_check,
            )

            return TestResult(
                case_id=case.id,
                description=case.description,
                passed=passed,
                answer=answer,
                missing_keywords=missing_keywords,
                citation_check_passed=citation_check,
            )

        except Exception as e:
            logger.error("TC %s raised exception: %s", case.id, e)
            return TestResult(
                case_id=case.id,
                description=case.description,
                passed=False,
                answer="",
                missing_keywords=list(case.expected_keywords),
                citation_check_passed=False,
                error=str(e),
            )

    def run_all(self) -> EvalReport:
        results = [self.run_single(tc) for tc in self.test_cases]
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pass_rate = passed / total if total > 0 else 0.0

        logger.info(
            "Eval complete: %d/%d passed (%.0f%%)", passed, total, pass_rate * 100
        )
        return EvalReport(
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=pass_rate,
            results=results,
        )

    def format_report(self, report: EvalReport) -> str:
        lines = [
            f"## Evaluation Results: {report.passed}/{report.total} passed "
            f"({report.pass_rate:.0%})\n",
            "| Test | Description | Result | Notes |",
            "|------|-------------|--------|-------|",
        ]

        for r in report.results:
            icon = "✅" if r.passed else "❌"
            if r.error:
                notes = f"Error: {r.error[:80]}"
            elif r.missing_keywords:
                notes = f"Missing keywords: {', '.join(r.missing_keywords)}"
            elif not r.citation_check_passed:
                notes = "No source citation found in answer"
            else:
                notes = "All checks passed"
            lines.append(f"| {r.case_id} | {r.description} | {icon} | {notes} |")

        lines.append(
            f"\n**Pass rate: {report.pass_rate:.0%}** "
            f"({report.passed}/{report.total} test cases)"
        )
        return "\n".join(lines)
