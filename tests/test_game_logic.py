from logic_utils import check_guess, parse_guess, update_score

# NOTE: check_guess returns (outcome, message) tuple.
# The three tests below unpack correctly.

def test_winning_guess():
    outcome, _ = check_guess(50, 50)
    assert outcome == "Win"

def test_guess_too_high():
    outcome, _ = check_guess(60, 50)
    assert outcome == "Too High"

def test_guess_too_low():
    outcome, _ = check_guess(40, 50)
    assert outcome == "Too Low"


# --- Tests targeting bugs that were fixed ---

# Bug: check_guess direction messages were swapped
# ("Too High" said "Go HIGHER!", "Too Low" said "Go LOWER!")
def test_too_high_message_says_go_lower():
    outcome, message = check_guess(60, 50)
    assert outcome == "Too High"
    assert "LOWER" in message

def test_too_low_message_says_go_higher():
    outcome, message = check_guess(40, 50)
    assert outcome == "Too Low"
    assert "HIGHER" in message


# Bug: parse_guess accepted negative numbers (no range check)
def test_parse_guess_rejects_negative():
    ok, _, _ = parse_guess("-1", 1, 100)
    assert ok is False

def test_parse_guess_rejects_above_range():
    ok, _, _ = parse_guess("101", 1, 100)
    assert ok is False

def test_parse_guess_accepts_valid():
    ok, value, _ = parse_guess("50", 1, 100)
    assert ok is True
    assert value == 50


# Bug: update_score gave +5 points for "Too High" on even attempts
# (rewarding a wrong guess). Should always be -5.
def test_update_score_too_high_even_attempt_deducts():
    score = update_score(100, "Too High", attempt_number=2)
    assert score == 95  # was 105 before fix

def test_update_score_too_high_odd_attempt_deducts():
    score = update_score(100, "Too High", attempt_number=3)
    assert score == 95

def test_update_score_too_low_deducts():
    score = update_score(100, "Too Low", attempt_number=2)
    assert score == 95
