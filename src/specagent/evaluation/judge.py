"""Answer correctness checking and LLM-as-judge evaluation."""

from specagent.llm.factory import get_llm

_JUDGE_PROMPT = (
    "You are evaluating answers to technical questions about 3GPP specifications.\n\n"
    "Question: Does the generated answer convey the same information as the expected answer?\n\n"
    "Expected Answer: {expected}\n"
    "Generated Answer: {generated}\n\n"
    'Respond with ONLY "yes" or "no".\n\n'
    "If the generated answer contains the expected answer or conveys the same key information "
    '(even with additional context), respond "yes".\n'
    "If the generated answer is incorrect, contradicts the expected answer, or is missing "
    'the key information, respond "no".\n\n'
    "Response:"
)


def _fuzzy_match(generated_norm: str, expected_norm: str) -> bool:
    """Return True if generated contains expected via exact, substring, or word-set match."""
    if generated_norm == expected_norm:
        return True
    if expected_norm in generated_norm:
        return True
    expected_words = set(expected_norm.split())
    generated_words = set(generated_norm.split())
    return bool(expected_words) and expected_words.issubset(generated_words)


def check_answer_correctness(
    generated: str,
    expected: str,
    use_llm_judge: bool = True,
) -> bool:
    """
    Check if generated answer matches expected answer.

    Uses fuzzy matching since exact string match is too strict.
    Falls back to LLM-as-judge for semantic comparison.

    Args:
        generated: Generated answer text
        expected: Expected/ground truth answer
        use_llm_judge: Whether to use LLM for semantic comparison

    Returns:
        True if answers match, False otherwise
    """
    if _fuzzy_match(generated.strip().lower(), expected.strip().lower()):
        return True
    if use_llm_judge:
        return llm_judge_answer(generated, expected)
    return False


def llm_judge_answer(generated: str, expected: str) -> bool:
    """
    Use LLM to judge if generated answer is semantically equivalent to expected.

    Args:
        generated: Generated answer
        expected: Expected answer

    Returns:
        True if answers are semantically equivalent
    """

    try:
        llm = get_llm()
        response = llm.invoke(_JUDGE_PROMPT.format(generated=generated, expected=expected))
        return "yes" in str(response).lower().strip()
    except Exception:
        return expected.lower() in generated.lower()
