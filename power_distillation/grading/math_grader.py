"""Math answer checker using normalization and sympy equality checks."""

import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

from power_distillation.grading import math_normalize

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    return re.compile("([0-9]) +([0-9])").sub("\\1+\\2", step)


def _strip_properly_formatted_commas(expr: str):
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = pattern.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    if expr is None:
        return None
    match = re.search(r"^\\\\text\{(?P<text>.+?)\}$", expr)
    if match is not None:
        expr = match.group("text")
    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute", "hour",
        "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "").replace("{", "").replace("}", "").lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "").replace("frac", "")
    letters_in_expr = {x for x in expr if x.isalpha()}
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False
    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False
    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False
    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            return sympy.simplify(sympy_diff) == 0
    except Exception:
        return False
    return False


def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)
    ):
        return [elem.strip() for elem in expr[1:-1].split(",")]
    return [expr]


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    if given_answer is None:
        return False
    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)
    if len(ground_truth_elems) != len(given_elems):
        return False
    return all(
        gt == gv or are_equal_under_sympy(gt, gv)
        for gt, gv in zip(ground_truth_elems, given_elems)
    )
