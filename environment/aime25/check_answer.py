"""
Simple answer checker for AIME dataset.
AIME answers are always integers from 0 to 999.

Usage:
    from check_answer import check_answer, extract_answer

    # Extract answer from model response
    answer = extract_answer(model_response)

    # Check if answer is correct
    result = check_answer("42", "42")  # True

    # Or from command line:
    python check_answer.py "42" "42"
"""

import re
from typing import Union, Optional


def extract_answer(response: str) -> Optional[str]:
    """
    Extract the final answer from model response.

    Looks for answers in common formats:
    1. \\boxed{XXX} - LaTeX boxed format (primary)
    2. "the answer is XXX" - Natural language format
    3. Standalone number at the end of response

    Args:
        response: The model's response text

    Returns:
        Extracted answer string, or None if not found
    """
    if not response:
        return None

    # Pattern 1: \boxed{XXX} (primary format)
    matches = re.findall(r'\\boxed\{(\d+)\}', response)
    if matches:
        return matches[-1].strip()

    # Pattern 2: The answer is XXX
    matches = re.findall(r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(\d+)', response, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    # Pattern 3: Look for standalone 3-digit or less number near the end
    lines = response.strip().split('\n')
    for line in reversed(lines[-5:]):
        line = line.strip()
        match = re.search(r'\b(\d{1,3})\b', line)
        if match:
            num = int(match.group(1))
            if 0 <= num <= 999:
                return str(num)

    return None


def extract_number(s: str) -> Optional[int]:
    """Extract integer from string."""
    if s is None:
        return None

    s = str(s).strip()

    # Remove common wrappers
    s = s.replace("$", "").replace("\\boxed{", "").replace("}", "")
    s = s.replace(",", "")  # Remove commas from numbers like 1,000

    # Try direct integer conversion
    try:
        return int(float(s))
    except:
        pass

    # Try to find a number in the string
    match = re.search(r"-?\d+", s)
    if match:
        try:
            return int(match.group())
        except:
            pass

    return None


def check_answer(
    prediction: Union[int, float, str],
    reference: Union[int, float, str],
) -> bool:
    """
    Check if prediction matches reference answer.

    For AIME, answers are integers from 0 to 999.

    Args:
        prediction: The predicted answer
        reference: The ground truth answer

    Returns:
        bool: True if answers match, False otherwise

    Examples:
        >>> check_answer("42", "42")
        True
        >>> check_answer("042", "42")
        True
        >>> check_answer(" 123 ", "123")
        True
        >>> check_answer("100", "99")
        False
    """
    pred_num = extract_number(prediction)
    ref_num = extract_number(reference)

    if pred_num is None or ref_num is None:
        # Fallback to string comparison
        pred_str = str(prediction).strip().lower()
        ref_str = str(reference).strip().lower()
        return pred_str == ref_str

    return pred_num == ref_num


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        pred = sys.argv[1]
        ref = sys.argv[2]
        result = check_answer(pred, ref)
        print(f"Prediction: {pred}")
        print(f"Reference:  {ref}")
        print(f"Match: {result}")
    else:
        # Run tests
        test_cases = [
            ("42", "42", True),
            ("042", "42", True),
            (" 123 ", "123", True),
            ("0", "0", True),
            ("999", "999", True),
            ("100", "99", False),
            ("1", "2", False),
            ("$42$", "42", True),
            ("\\boxed{123}", "123", True),
            ("The answer is 456", "456", True),
            ("1,000", "1000", True),
            ("-5", "-5", True),
        ]

        print("Running tests...")
        passed = 0
        for pred, ref, expected in test_cases:
            result = check_answer(pred, ref)
            status = "✓" if result == expected else "✗"
            if result == expected:
                passed += 1
            print(f"{status} check_answer(\"{pred}\", \"{ref}\") = {result} (expected: {expected})")

        print(f"\nResult: {passed}/{len(test_cases)} passed")
