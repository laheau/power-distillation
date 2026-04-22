"""Helpers to extract boxed answers from model completions."""


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def _boxed_only_string(string, reverse=False):
    search_terms = ["\\boxed", "\\fbox"]
    candidates = []
    for term in search_terms:
        start = 0
        while True:
            idx = string.find(term, start)
            if idx < 0:
                break
            candidates.append(idx)
            start = idx + 1

    if not candidates:
        return None

    ordered = sorted(candidates, reverse=reverse)
    for idx in ordered:
        i = idx
        while i < len(string) and string[i] != "{":
            i += 1
        if i >= len(string):
            continue
        right_brace_idx = None
        num_left_braces_open = 0
        j = i
        while j < len(string):
            if string[j] == "{":
                num_left_braces_open += 1
            if string[j] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = j
                    break
            j += 1
        if right_brace_idx is None:
            continue
        retval = string[idx : right_brace_idx + 1]
        content = remove_boxed(retval) if retval.startswith("\\boxed{") else retval
        if content is not None and content.strip().replace("{", "").replace("}", "").strip() != "":
            return retval
    return None


def first_boxed_only_string(string):
    return _boxed_only_string(string, reverse=False)


def parse_answer(input_str):
    return remove_boxed(first_boxed_only_string(input_str))

