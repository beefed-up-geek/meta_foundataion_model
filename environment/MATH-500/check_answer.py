"""
Math answer checker for MATH-500 dataset.
Supports diverse answer formats: fractions, radicals, complex numbers, coordinates,
angles, text, multiple answers, decimals, intervals, matrices, polynomials, etc.

Usage:
    from check_answer import check_answer
    result = check_answer("\\frac{1}{2}", "0.5")  # True

    # Or from command line:
    python check_answer.py "\\frac{1}{2}" "0.5"
"""

import re
import multiprocessing
from math import isclose, sqrt, pi
from typing import Union, Optional, List, Set
from fractions import Fraction

from sympy import simplify, N, sympify, Rational, sqrt as sym_sqrt, pi as sym_pi, I, oo
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy.parsing.latex import parse_latex

try:
    from latex2sympy2 import latex2sympy
except ImportError:
    latex2sympy = None


def extract_answer(response: str) -> Optional[str]:
    """
    Extract the final answer from model response.

    Looks for answers in common formats:
    1. \\boxed{...} - LaTeX boxed format (primary)
    2. "the answer is ..." - Natural language format
    3. "final answer: ..." - Alternative format

    Args:
        response: The model's response text

    Returns:
        Extracted answer string, or None if not found
    """
    if not response:
        return None

    # Pattern 1: \boxed{...} - Handle nested braces
    # Find all \boxed occurrences and extract content with balanced braces
    boxed_matches = []
    i = 0
    while i < len(response):
        if response[i:i+7] == '\\boxed{':
            start = i + 7
            depth = 1
            j = start
            while j < len(response) and depth > 0:
                if response[j] == '{':
                    depth += 1
                elif response[j] == '}':
                    depth -= 1
                j += 1
            if depth == 0:
                boxed_matches.append(response[start:j-1])
            i = j
        else:
            i += 1

    if boxed_matches:
        # Return the last boxed answer (most likely the final answer)
        return boxed_matches[-1].strip()

    # Pattern 2: "The answer is ..." or "The final answer is ..."
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)',
        r'(?:therefore|thus|hence)[,\s]+(?:the\s+)?(?:final\s+)?answer\s+is\s*[:\s]*(.+?)(?:\.|$)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            answer = matches[-1].strip()
            # Clean up the answer
            answer = re.sub(r'\s+', ' ', answer)
            if answer:
                return answer

    return None


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    if s is None:
        return ""
    s = str(s).strip()

    # Remove outer $ delimiters
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # Remove \boxed{...}
    boxed_match = re.match(r"^\\boxed\{(.+)\}$", s, re.DOTALL)
    if boxed_match:
        s = boxed_match.group(1).strip()

    # Remove \text{...} wrapper but keep content
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)

    # Remove \left and \right
    s = s.replace("\\left", "").replace("\\right", "")

    # Normalize \! (small space in LaTeX)
    s = s.replace("\\!", "")
    s = s.replace("\\,", "")
    s = s.replace("\\;", "")
    s = s.replace("\\ ", " ")

    # Normalize degree symbols
    s = re.sub(r"\^\s*\\circ", "", s)
    s = re.sub(r"\^\s*\{\\circ\}", "", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace("\\circ", "")

    # Normalize dfrac/tfrac to frac
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # Normalize sqrt: sqrt(x) -> \sqrt{x}
    s = re.sub(r"\bsqrt\(([^)]+)\)", r"\\sqrt{\1}", s)

    # Handle space variations in sqrt: 11\sqrt2 -> 11\sqrt{2}
    s = re.sub(r"(\\sqrt)(\d+)", r"\1{\2}", s)

    # Normalize pi variations
    s = re.sub(r"(?<!\\)\bpi\b", r"\\pi", s)

    # Normalize infinity
    s = s.replace("\\infty", "oo")

    # Remove spaces around operators for sqrt expressions like "70 \sqrt{2}" -> "70\sqrt{2}"
    s = re.sub(r"(\d)\s+(\\sqrt)", r"\1\2", s)

    return s.strip()


def extract_text_content(s: str) -> Optional[str]:
    """Extract text content from \\text{...} or similar."""
    s = str(s).strip()
    patterns = [
        r"^\\text\{([^}]+)\}$",
        r"^\\mbox\{([^}]+)\}$",
        r"^\\mathrm\{([^}]+)\}$",
    ]
    for pattern in patterns:
        match = re.match(pattern, s)
        if match:
            return match.group(1).strip().lower()
    return None


def extract_choice_letter(s: str) -> Optional[str]:
    """Extract choice letter (A-E) from various formats."""
    s = str(s).strip()
    # Remove \text{...} wrapper
    text_match = re.match(r"^\\text\{(.+)\}$", s)
    if text_match:
        s = text_match.group(1).strip()
    # Remove parentheses
    s = s.strip("()")
    # Check if it's a single letter A-E
    if len(s) == 1 and s.upper() in 'ABCDE':
        return s.upper()
    # Try to find any letter A-E in the string
    for char in s.upper():
        if char in 'ABCDE':
            return char
    return None


def is_text_answer(s: str) -> bool:
    """Check if answer is a text-based answer."""
    return extract_text_content(s) is not None or s.strip().isalpha()


def parse_latex_frac(s: str) -> Optional[float]:
    """Parse LaTeX fraction like \\frac{1}{2} to float."""
    s = str(s).strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # Simple pattern for \frac{num}{denom}
    frac_pattern = r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}"
    match = re.search(frac_pattern, s)
    if match:
        try:
            num_str = match.group(1).strip()
            denom_str = match.group(2).strip()
            num = float(num_str)
            denom = float(denom_str)
            if denom != 0:
                return num / denom
        except:
            pass
    return None


def parse_number(s: str) -> Optional[float]:
    """Parse a string to float, handling various formats."""
    if s is None:
        return None

    original = str(s).strip()

    # Try LaTeX fraction first (before removing braces)
    frac = parse_latex_frac(original)
    if frac is not None:
        return frac

    # Now clean up for other parsing
    s = original.replace("$", "")
    s = re.sub(r"\\boxed\{(.+)\}", r"\1", s)  # Remove \boxed{} properly
    s = s.replace(",", "")  # Remove commas
    s = s.replace(" ", "")  # Remove spaces

    # Handle percentages
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100
        except:
            pass

    # Try direct conversion
    try:
        return float(s)
    except:
        pass

    # Try using sympy for LaTeX parsing
    try:
        result = parse_latex(original)
        num_result = float(N(result))
        return num_result
    except:
        pass

    return None


def is_number(s: str) -> bool:
    """Check if string can be parsed as a number."""
    return parse_number(s) is not None


def numeric_equal(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    """Check if two numbers are equal within tolerance."""
    if a is None or b is None:
        return False
    return isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def parse_multiple_answers(s: str) -> Optional[Set[str]]:
    """Parse answers with multiple values like '1,-2' or '3, 5, 7'."""
    s = normalize_answer(s)

    # Check if it looks like multiple answers (contains comma but not in brackets/parens)
    if ',' not in s:
        return None

    # Don't treat coordinates/tuples as multiple answers
    if re.match(r"^\s*[\(\[\{].*[\)\]\}]\s*$", s):
        return None

    # Don't treat intervals as multiple answers
    if re.match(r"^\s*[\(\[].*,.*[\)\]]\s*$", s) and ('\\infty' in s or 'oo' in s or 'inf' in s.lower()):
        return None

    # Split by comma
    parts = [p.strip() for p in s.split(',')]
    if len(parts) > 1:
        return set(parts)
    return None


def compare_multiple_answers(pred: str, ref: str) -> bool:
    """Compare answers that may contain multiple values (order-independent)."""
    pred_set = parse_multiple_answers(pred)
    ref_set = parse_multiple_answers(ref)

    if pred_set is None or ref_set is None:
        return False

    if len(pred_set) != len(ref_set):
        return False

    # Try to match each element
    matched_ref = set()
    for p in pred_set:
        for r in ref_set:
            if r not in matched_ref:
                if check_single_answer(p, r):
                    matched_ref.add(r)
                    break

    return len(matched_ref) == len(ref_set)


def parse_coordinate(s: str) -> Optional[List[str]]:
    """Parse coordinate/tuple like (1, 2) or (3, \\frac{\\pi}{2})."""
    s = normalize_answer(s)

    # Match (a, b) or (a, b, c) etc.
    match = re.match(r"^\s*[\(\[]\s*(.+)\s*[\)\]]\s*$", s)
    if match:
        inner = match.group(1)
        # Split by comma, but be careful with nested structures
        parts = []
        depth = 0
        current = ""
        for char in inner:
            if char in '([{':
                depth += 1
                current += char
            elif char in ')]}':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            parts.append(current.strip())
        return parts if len(parts) > 1 else None
    return None


def compare_coordinates(pred: str, ref: str) -> bool:
    """Compare coordinate answers."""
    pred_parts = parse_coordinate(pred)
    ref_parts = parse_coordinate(ref)

    if pred_parts is None or ref_parts is None:
        return False

    if len(pred_parts) != len(ref_parts):
        return False

    return all(check_single_answer(p, r) for p, r in zip(pred_parts, ref_parts))


def is_interval(s: str) -> bool:
    """Check if string looks like an interval (has infinity or mixed brackets)."""
    s_lower = s.lower()
    # Contains infinity
    if 'infty' in s_lower or 'oo' in s_lower or 'inf' in s_lower:
        return True
    # Has mixed brackets like [a, b) or (a, b]
    if (s.startswith('[') and s.endswith(')')) or (s.startswith('(') and s.endswith(']')):
        return True
    return False


def parse_interval(s: str) -> Optional[tuple]:
    """Parse interval notation like (2, \\infty) or [0, 1)."""
    s = normalize_answer(s)

    # Only parse as interval if it looks like one
    if not is_interval(s):
        return None

    # Match interval pattern
    match = re.match(r"^\s*([\(\[])\s*(.+?)\s*,\s*(.+?)\s*([\)\]])\s*$", s)
    if match:
        left_bracket = match.group(1)
        left_val = match.group(2).strip()
        right_val = match.group(3).strip()
        right_bracket = match.group(4)
        return (left_bracket, left_val, right_val, right_bracket)
    return None


def compare_intervals(pred: str, ref: str) -> bool:
    """Compare interval answers."""
    pred_interval = parse_interval(pred)
    ref_interval = parse_interval(ref)

    if pred_interval is None or ref_interval is None:
        return False

    # Check brackets match
    if pred_interval[0] != ref_interval[0] or pred_interval[3] != ref_interval[3]:
        return False

    # Check endpoints
    return (check_single_answer(pred_interval[1], ref_interval[1]) and
            check_single_answer(pred_interval[2], ref_interval[2]))


def parse_complex(s: str) -> Optional[complex]:
    """Parse complex number like '6 - 5i' or '1+2i'."""
    s = normalize_answer(s)
    s = s.replace(" ", "")

    # Pattern for a + bi or a - bi
    patterns = [
        r"^(-?\d*\.?\d*)\s*([+-])\s*(\d*\.?\d*)i$",  # a +/- bi
        r"^(-?\d*\.?\d*)i\s*([+-])\s*(\d*\.?\d*)$",  # ai +/- b (less common)
        r"^(-?\d*\.?\d*)i$",  # bi only
        r"^(-?\d*\.?\d*)$",  # a only (real)
    ]

    # a +/- bi
    match = re.match(r"^(-?\d*\.?\d*)\s*([+-])\s*(\d*\.?\d*)i$", s)
    if match:
        try:
            real = float(match.group(1)) if match.group(1) else 0
            sign = 1 if match.group(2) == '+' else -1
            imag = float(match.group(3)) if match.group(3) else 1
            return complex(real, sign * imag)
        except:
            pass

    # bi only
    match = re.match(r"^(-?\d*\.?\d*)i$", s)
    if match:
        try:
            imag = float(match.group(1)) if match.group(1) and match.group(1) != '-' else (1 if not match.group(1) else -1)
            return complex(0, imag)
        except:
            pass

    return None


def compare_complex(pred: str, ref: str) -> bool:
    """Compare complex number answers."""
    pred_c = parse_complex(pred)
    ref_c = parse_complex(ref)

    if pred_c is None or ref_c is None:
        return False

    return (numeric_equal(pred_c.real, ref_c.real) and
            numeric_equal(pred_c.imag, ref_c.imag))


def parse_base_number(s: str) -> Optional[tuple]:
    """Parse number in different base like 4210_5."""
    s = normalize_answer(s)
    match = re.match(r"^(\d+)_\{?(\d+)\}?$", s)
    if match:
        return (match.group(1), int(match.group(2)))
    return None


def compare_base_numbers(pred: str, ref: str) -> bool:
    """Compare numbers in different bases."""
    pred_base = parse_base_number(pred)
    ref_base = parse_base_number(ref)

    if pred_base is None or ref_base is None:
        return False

    # Must be same base
    if pred_base[1] != ref_base[1]:
        return False

    # Compare the digits
    return pred_base[0] == ref_base[0]


def parse_polynomial(s: str) -> Optional[str]:
    """Normalize polynomial expression."""
    s = normalize_answer(s)
    # Remove spaces
    s = s.replace(" ", "")
    return s if re.search(r'[a-z]\^?\d*', s) else None


def compare_polynomials(pred: str, ref: str) -> bool:
    """Compare polynomial expressions using sympy."""
    try:
        pred_norm = normalize_answer(pred).replace("^", "**")
        ref_norm = normalize_answer(ref).replace("^", "**")

        pred_expr = parse_expr(pred_norm, transformations=standard_transformations + (implicit_multiplication_application,))
        ref_expr = parse_expr(ref_norm, transformations=standard_transformations + (implicit_multiplication_application,))

        diff = simplify(pred_expr - ref_expr)
        return diff == 0
    except:
        return False


def symbolic_equal(a: str, b: str) -> bool:
    """Check if two expressions are symbolically equal using sympy."""

    def _parse(s):
        """Try multiple parsers."""
        s = s.replace("\\\\", "\\")

        # Replace ^ with ** for sympy
        s_power = s.replace("^", "**")

        # Try latex2sympy first if available
        parsers = []
        if latex2sympy is not None:
            parsers.append(lambda x: latex2sympy(x))
        parsers.extend([
            lambda x: parse_latex(x),
            lambda x: parse_expr(x, transformations=standard_transformations + (implicit_multiplication_application,)),
            lambda x: parse_expr(x.replace("^", "**"), transformations=standard_transformations + (implicit_multiplication_application,)),
        ])

        for parser in parsers:
            try:
                result = parser(s)
                if result is not None:
                    return result
            except:
                pass
            try:
                result = parser(s_power)
                if result is not None:
                    return result
            except:
                pass
        return None

    a_parsed = _parse(a)
    b_parsed = _parse(b)

    if a_parsed is None or b_parsed is None:
        return False

    # Direct equality
    try:
        if a_parsed == b_parsed:
            return True
        if str(a_parsed) == str(b_parsed):
            return True
    except:
        pass

    # Simplify difference
    try:
        if simplify(a_parsed - b_parsed) == 0:
            return True
    except:
        pass

    # Use .equals method
    try:
        if a_parsed.equals(b_parsed):
            return True
    except:
        pass

    # Numeric evaluation
    try:
        a_num = complex(N(a_parsed))
        b_num = complex(N(b_parsed))
        if numeric_equal(a_num.real, b_num.real) and numeric_equal(a_num.imag, b_num.imag):
            return True
    except:
        pass

    return False


def check_single_answer(prediction: str, reference: str) -> bool:
    """Check if a single prediction matches a single reference."""
    if prediction is None or reference is None:
        return False

    pred = normalize_answer(str(prediction))
    ref = normalize_answer(str(reference))

    # Empty check
    if not pred or not ref:
        return pred == ref

    # 1. Exact string match
    if pred.lower() == ref.lower():
        return True

    # 2. Multiple choice (A, B, C, D, E) - check before text comparison
    pred_choice = extract_choice_letter(prediction)
    ref_choice = extract_choice_letter(reference)
    if pred_choice is not None and ref_choice is not None:
        return pred_choice == ref_choice
    # Also handle case where one is extracted and the other is plain
    if pred_choice is not None:
        ref_clean = re.sub(r'[^A-E]', '', ref.upper())
        if ref_clean:
            return pred_choice == ref_clean[-1]
    if ref_choice is not None:
        pred_clean = re.sub(r'[^A-E]', '', pred.upper())
        if pred_clean:
            return ref_choice == pred_clean[-1]

    # 3. Text answer comparison
    pred_text = extract_text_content(prediction)
    ref_text = extract_text_content(reference)
    if pred_text is not None and ref_text is not None:
        return pred_text.lower() == ref_text.lower()
    if pred_text is not None:
        return pred_text.lower() == ref.lower()
    if ref_text is not None:
        return pred.lower() == ref_text.lower()

    # 4. Numeric comparison
    pred_num = parse_number(pred)
    ref_num = parse_number(ref)
    if pred_num is not None and ref_num is not None:
        return numeric_equal(pred_num, ref_num)

    # 5. Complex number comparison
    if 'i' in pred.lower() and 'i' in ref.lower():
        if compare_complex(pred, ref):
            return True

    # 6. Base number comparison (e.g., 4210_5)
    if '_' in pred and '_' in ref:
        if compare_base_numbers(pred, ref):
            return True

    # 7. Symbolic/algebraic comparison
    try:
        if symbolic_equal(pred, ref):
            return True
    except:
        pass

    return False


def check_answer(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Check if prediction matches reference answer.

    Supports:
    - Integers and decimals
    - Fractions (\\frac{a}{b})
    - Radicals (\\sqrt{n}, 3\\sqrt{13})
    - Complex numbers (6 - 5i)
    - Coordinates ((3, 4), (x, y, z))
    - Intervals ((2, \\infty), [0, 1])
    - Multiple answers (1,-2 vs -2,1)
    - Angles (90^\\circ)
    - Text answers (\\text{Evelyn})
    - Polynomials (x^2 + 2x + 1)
    - Matrices (\\begin{pmatrix}...\\end{pmatrix})
    - Base numbers (4210_5)
    - Currency ($32,348)

    Args:
        prediction: The predicted answer
        reference: The ground truth answer
        include_percentage: Whether to consider percentage variations
        is_close: Whether to use approximate numeric comparison
        timeout: Whether to use timeout for symbolic comparison

    Returns:
        bool: True if answers match, False otherwise
    """
    if prediction is None or reference is None:
        return False

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    # Quick exact match
    if prediction == reference:
        return True

    pred_norm = normalize_answer(prediction)
    ref_norm = normalize_answer(reference)

    if pred_norm.lower() == ref_norm.lower():
        return True

    # 1. Try single answer comparison first
    if check_single_answer(prediction, reference):
        return True

    # 2. Interval comparison (check before coordinates to avoid confusion)
    pred_is_interval = is_interval(prediction) or is_interval(normalize_answer(prediction))
    ref_is_interval = is_interval(reference) or is_interval(normalize_answer(reference))
    if pred_is_interval or ref_is_interval:
        if compare_intervals(prediction, reference):
            return True
        # If one is interval and comparison failed, they're different
        if pred_is_interval and ref_is_interval:
            return False

    # 3. Multiple answers (order-independent)
    if ',' in prediction or ',' in reference:
        if compare_multiple_answers(prediction, reference):
            return True

    # 4. Coordinate/tuple comparison
    if ('(' in prediction or '[' in prediction) and ('(' in reference or '[' in reference):
        if compare_coordinates(prediction, reference):
            return True

    # 5. Matrix comparison
    if 'pmatrix' in prediction.lower() or 'bmatrix' in prediction.lower():
        if compare_matrices(prediction, reference):
            return True

    # 6. Equation comparison (x = 5)
    if '=' in prediction and '=' in reference:
        if compare_equations(prediction, reference):
            return True

    # Handle x=5 vs 5
    if '=' in prediction and '=' not in reference:
        parts = prediction.split('=')
        if len(parts) == 2 and len(parts[0].strip()) <= 2:
            if check_single_answer(parts[1].strip(), reference):
                return True
    if '=' not in prediction and '=' in reference:
        parts = reference.split('=')
        if len(parts) == 2 and len(parts[0].strip()) <= 2:
            if check_single_answer(prediction, parts[1].strip()):
                return True

    # 7. Polynomial comparison
    if re.search(r'[a-z]\^?\d*', pred_norm) and re.search(r'[a-z]\^?\d*', ref_norm):
        if compare_polynomials(prediction, reference):
            return True

    return False


def compare_matrices(pred: str, ref: str) -> bool:
    """Compare matrix answers."""
    def extract_matrix(s):
        s = normalize_answer(s)
        # Find matrix content
        match = re.search(r"\\begin\{[pb]matrix\}(.+?)\\end\{[pb]matrix\}", s, re.DOTALL)
        if match:
            content = match.group(1)
            rows = content.split("\\\\")
            matrix = []
            for row in rows:
                if row.strip():
                    cells = [c.strip() for c in row.split("&")]
                    matrix.append(cells)
            return matrix
        return None

    pred_mat = extract_matrix(pred)
    ref_mat = extract_matrix(ref)

    if pred_mat is None or ref_mat is None:
        return False

    if len(pred_mat) != len(ref_mat):
        return False

    for pred_row, ref_row in zip(pred_mat, ref_mat):
        if len(pred_row) != len(ref_row):
            return False
        for p, r in zip(pred_row, ref_row):
            if not check_single_answer(p, r):
                return False
    return True


def compare_equations(pred: str, ref: str) -> bool:
    """Compare equation answers."""
    try:
        pred_parts = pred.split('=')
        ref_parts = ref.split('=')

        if len(pred_parts) != 2 or len(ref_parts) != 2:
            return False

        # Create expressions: lhs - rhs
        pred_expr = f"({pred_parts[0].strip()}) - ({pred_parts[1].strip()})"
        ref_expr = f"({ref_parts[0].strip()}) - ({ref_parts[1].strip()})"

        if symbolic_equal(pred_expr, ref_expr):
            return True
        # Try negation
        if symbolic_equal(f"-({pred_expr})", ref_expr):
            return True
    except:
        pass
    return False


# Alias for backward compatibility
math_equal = check_answer


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        pred = sys.argv[1]
        ref = sys.argv[2]
        result = check_answer(pred, ref, timeout=True)
        print(f"Prediction: {pred}")
        print(f"Reference:  {ref}")
        print(f"Match: {result}")
    else:
        # Run comprehensive tests
        test_cases = [
            # Basic numbers
            ("42", "42", True),
            ("0.5", "0.5", True),
            ("-3", "-3", True),

            # Fractions
            ("0.5", "\\frac{1}{2}", True),
            ("\\frac{14}{3}", "\\frac{14}{3}", True),
            ("\\dfrac{17}{50}", "\\frac{17}{50}", True),

            # Decimals
            ("1.25", "1.25", True),
            (".35625", "0.35625", True),
            ("326.5", "326.5", True),

            # Radicals
            ("\\sqrt{51}", "\\sqrt{51}", True),
            ("3\\sqrt{13}", "3\\sqrt{13}", True),
            ("11\\sqrt2", "11\\sqrt{2}", True),
            ("70 \\sqrt{2}", "70\\sqrt{2}", True),

            # Complex numbers
            ("6 - 5i", "6-5i", True),
            ("1+274i", "1 + 274i", True),
            ("-2 + 7i", "-2+7i", True),

            # Coordinates
            ("(3, 4)", "(3,4)", True),
            ("(6,31,-1)", "(6, 31, -1)", True),

            # Angles (without degree symbol after normalization)
            ("90", "90", True),
            ("145", "145", True),

            # Text answers
            ("\\text{Evelyn}", "\\text{Evelyn}", True),
            ("\\text{even}", "\\text{even}", True),
            ("Evelyn", "\\text{Evelyn}", True),

            # Multiple choice
            ("A", "A", True),
            ("\\text{(C)}", "(C)", True),
            ("(E)", "\\text{(E)}", True),

            # Multiple answers (order-independent)
            ("1,-2", "-2,1", True),
            ("-2, 1", "1,-2", True),
            ("3, 5, 7", "5, 3, 7", True),

            # Intervals
            ("(2,\\infty)", "(2, \\infty)", True),
            ("(3,4]", "(3, 4]", True),
            ("(-\\infty, 0]", "(-\\infty,0]", True),

            # Base numbers
            ("4210_{5}", "4210_5", True),

            # Polynomials
            ("x^3+3x-6", "x^3 + 3x - 6", True),
            ("x+2", "2+x", True),

            # Equations
            ("x=5", "x = 5", True),
            ("y = -2x", "y=-2x", True),

            # Matrices
            ("\\begin{pmatrix} -1 & 0 \\\\ 0 & -1 \\end{pmatrix}",
             "\\begin{pmatrix}-1&0\\\\0&-1\\end{pmatrix}", True),

            # Currency
            ("32348", "32,348", True),

            # Pi expressions
            ("\\pi", "\\pi", True),
            ("12\\pi", "12\\pi", True),

            # Wrong answers
            ("42", "43", False),
            ("A", "B", False),
            ("(1,2)", "(1,3)", False),
        ]

        print("Running comprehensive tests...")
        passed = 0
        failed = []
        for pred, ref, expected in test_cases:
            result = check_answer(pred, ref)
            if result == expected:
                passed += 1
                status = "PASS"
            else:
                status = "FAIL"
                failed.append((pred, ref, expected, result))
            print(f"{status}: check_answer(\"{pred}\", \"{ref}\") = {result} (expected: {expected})")

        print(f"\n{'='*60}")
        print(f"Result: {passed}/{len(test_cases)} passed")

        if failed:
            print(f"\nFailed cases:")
            for pred, ref, expected, result in failed:
                print(f"  - \"{pred}\" vs \"{ref}\": got {result}, expected {expected}")
