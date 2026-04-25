import logging
from dataclasses import dataclass
from typing import Optional

from strategy_rag import StrategyRAG

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    id: str
    situation: str
    question: str
    expected_keywords: list
    description: str


@dataclass
class TestResult:
    case_id: str
    description: str
    passed: bool
    advice: str
    missing_keywords: list
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
        situation="I've been guessing randomly with no strategy.",
        question="What strategy should I use to guess the number?",
        expected_keywords=["binary", "search", "midpoint", "half"],
        description="Random guesser needs binary search advice",
    ),
    TestCase(
        id="TC02",
        situation="Secret is between 20 and 80. I guessed 50 and it was too low.",
        question="What should I guess next when my guess was too low?",
        expected_keywords=["higher", "above", "51"],
        description="Directional narrowing after too-low feedback",
    ),
    TestCase(
        id="TC03",
        situation="I've used 7 of my 8 attempts and still haven't found it.",
        question="What should I do with my last remaining guess?",
        expected_keywords=["narrow", "remain", "careful"],
        description="Urgency tactics with one attempt remaining",
    ),
    TestCase(
        id="TC04",
        situation="I guessed 1 as my very first guess in a 1-100 game.",
        question="Was guessing 1 first a good opening move?",
        expected_keywords=["midpoint", "50", "middle"],
        description="Critique of inefficient opening guess",
    ),
    TestCase(
        id="TC05",
        situation="Starting a new 1-100 game with no information yet.",
        question="What is the best first guess for a 1-100 range?",
        expected_keywords=["50", "midpoint", "binary"],
        description="Optimal first-guess recommendation",
    ),
]


class StrategyEvalSuite:

    def __init__(self, rag: StrategyRAG):
        self.rag = rag
        self.test_cases = BUILT_IN_TEST_CASES

    def run_single(self, case: TestCase) -> TestResult:
        try:
            response = self.rag.advise(case.question)
            advice = response.advice
            advice_lower = advice.lower()

            missing_keywords = [
                kw for kw in case.expected_keywords if kw.lower() not in advice_lower
            ]
            passed = len(missing_keywords) == 0

            logger.info(
                "TC %s (%s): %s | missing=%s",
                case.id,
                case.description,
                "PASS" if passed else "FAIL",
                missing_keywords,
            )

            return TestResult(
                case_id=case.id,
                description=case.description,
                passed=passed,
                advice=advice,
                missing_keywords=missing_keywords,
            )

        except Exception as e:
            logger.error("TC %s raised exception: %s", case.id, e)
            return TestResult(
                case_id=case.id,
                description=case.description,
                passed=False,
                advice="",
                missing_keywords=list(case.expected_keywords),
                error=str(e),
            )

    def run_all(self) -> EvalReport:
        results = [self.run_single(tc) for tc in self.test_cases]
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pass_rate = passed / total if total > 0 else 0.0

        logger.info(
            "Strategy eval complete: %d/%d passed (%.0f%%)", passed, total, pass_rate * 100
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
            f"## Strategy Evaluation: {report.passed}/{report.total} passed "
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
            else:
                notes = "All keywords found"
            lines.append(f"| {r.case_id} | {r.description} | {icon} | {notes} |")

        lines.append(
            f"\n**Pass rate: {report.pass_rate:.0%}** "
            f"({report.passed}/{report.total} test cases)"
        )
        return "\n".join(lines)
