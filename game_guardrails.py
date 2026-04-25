MIN_QUERY_LENGTH = 5
MAX_QUERY_LENGTH = 500

DIRECTION_KEYWORDS_LOWER = frozenset({"lower", "less", "down", "smaller", "below", "decrease"})
DIRECTION_KEYWORDS_HIGHER = frozenset({"higher", "more", "up", "larger", "above", "increase"})


class GameGuardrailError(Exception):
    pass


class EmptyQueryError(GameGuardrailError):
    pass


class QueryTooShortError(GameGuardrailError):
    pass


class QueryTooLongError(GameGuardrailError):
    pass


class InvalidHintOutputError(GameGuardrailError):
    pass


class QueryValidator:

    def validate_query(self, text: str) -> None:
        stripped = text.strip()
        if not stripped:
            raise EmptyQueryError("Query cannot be empty.")
        if len(stripped) < MIN_QUERY_LENGTH:
            raise QueryTooShortError(
                f"Query too short (minimum {MIN_QUERY_LENGTH} characters)."
            )
        if len(stripped) > MAX_QUERY_LENGTH:
            raise QueryTooLongError(
                f"Query too long (maximum {MAX_QUERY_LENGTH} characters)."
            )

    def validate_hint_output(self, hint: str, outcome: str) -> None:
        hint_lower = hint.lower()
        if outcome == "Too High":
            if not any(kw in hint_lower for kw in DIRECTION_KEYWORDS_LOWER):
                raise InvalidHintOutputError(
                    f"Hint for 'Too High' must contain a direction keyword "
                    f"(e.g. 'lower', 'less', 'down'). Got: {hint!r}"
                )
        elif outcome == "Too Low":
            if not any(kw in hint_lower for kw in DIRECTION_KEYWORDS_HIGHER):
                raise InvalidHintOutputError(
                    f"Hint for 'Too Low' must contain a direction keyword "
                    f"(e.g. 'higher', 'more', 'up'). Got: {hint!r}"
                )

    def sanitize_query(self, text: str) -> str:
        return text.strip()[:MAX_QUERY_LENGTH]
