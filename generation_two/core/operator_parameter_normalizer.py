"""
Operator parameter normalization for WorldQuant FASTEXPR.

WorldQuant distinguishes expression inputs from operator attributes. For
example, ts_sum(x, d) needs a numeric lookback, while winsorize(x, std=4)
should not receive std as an unnamed second input.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


LOOKBACK_ARGUMENTS = {
    "d",
    "days",
    "lookback",
    "window",
    "period",
    "lagperiod",
}


def split_top_level_args(text: str) -> List[str]:
    """Split a comma-separated argument list, ignoring commas in nested calls."""
    args = []
    current = []
    depth = 0

    for char in text:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            arg = "".join(current).strip()
            if arg:
                args.append(arg)
            current = []
        else:
            current.append(char)

    arg = "".join(current).strip()
    if arg:
        args.append(arg)

    return args


def _definition_args(definition: str) -> List[str]:
    match = re.search(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(([^)]*)\)", definition or "")
    if not match:
        return []
    return split_top_level_args(match.group(1))


def _clean_argument_name(arg: str) -> str:
    arg = arg.split("=", 1)[0]
    arg = re.sub(r"\.\.+", "", arg)
    arg = re.sub(r"[^a-zA-Z0-9_]", "", arg)
    return arg.strip().lower()


def required_argument_names(definition: str) -> List[str]:
    """Return required positional argument names from an operator definition."""
    names = []
    for arg in _definition_args(definition):
        if "=" in arg:
            break
        name = _clean_argument_name(arg)
        if name:
            names.append(name)
    return names


def required_argument_count(definition: str) -> int:
    return len(required_argument_names(definition))


def _has_named_optional_args(definition: str) -> bool:
    return any("=" in arg for arg in _definition_args(definition))


def _is_variadic(definition: str) -> bool:
    text = (definition or "").lower()
    return ".." in text or "..." in text or "at least" in text or "minimum" in text


def _default_for_missing_arg(arg_name: str) -> str:
    if arg_name.lower() in LOOKBACK_ARGUMENTS:
        return "20"
    return ""


def _find_function_calls(expression: str) -> List[Tuple[int, int, str, str]]:
    calls = []
    pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")

    for match in pattern.finditer(expression):
        name = match.group(1)
        open_pos = expression.find("(", match.start())
        if open_pos < 0:
            continue

        depth = 0
        for pos in range(open_pos, len(expression)):
            if expression[pos] == "(":
                depth += 1
            elif expression[pos] == ")":
                depth -= 1
                if depth == 0:
                    args_text = expression[open_pos + 1:pos]
                    calls.append((match.start(), pos + 1, name, args_text))
                    break

    return calls


def normalize_operator_parameters(expression: str, operators: List[Dict]) -> Tuple[str, List[str]]:
    """
    Normalize obvious operator parameter mistakes after placeholders are resolved.

    - Removes unnamed optional attributes, e.g. winsorize(x, 10) -> winsorize(x)
    - Preserves required lookbacks, e.g. ts_sum(x, 20)
    - Adds missing lookbacks when the definition makes them required, e.g.
      ts_sum(x) -> ts_sum(x, 20)
    """
    if not expression or not operators:
        return expression, []

    definitions = {
        str(op.get("name", "")).lower(): str(op.get("definition", ""))
        for op in operators
        if op.get("name")
    }

    fixed = expression
    fixes: List[str] = []

    # Re-scan after every change so nested span offsets never go stale.
    for _ in range(100):
        changed = False
        calls = _find_function_calls(fixed)

        for start, end, op_name, args_text in reversed(calls):
            definition = definitions.get(op_name.lower())
            if not definition:
                continue

            required_names = required_argument_names(definition)
            required_count = len(required_names)
            if required_count == 0:
                continue

            args = split_top_level_args(args_text)
            new_args = list(args)

            if (
                len(new_args) > required_count
                and not _is_variadic(definition)
                and _has_named_optional_args(definition)
            ):
                optional_args = new_args[required_count:]
                kept_named = [arg for arg in optional_args if "=" in arg]
                removed = [arg for arg in optional_args if "=" not in arg]
                if removed:
                    new_args = new_args[:required_count] + kept_named
                    fixes.append(
                        f"Removed unnamed optional parameter(s) from {op_name}: {', '.join(removed)}"
                    )

            if len(new_args) < required_count:
                missing_names = required_names[len(new_args):]
                defaults = [_default_for_missing_arg(name) for name in missing_names]
                if defaults and all(defaults):
                    new_args.extend(defaults)
                    fixes.append(
                        f"Added default lookback parameter(s) to {op_name}: {', '.join(defaults)}"
                    )

            if new_args != args:
                replacement = f"{op_name}({', '.join(new_args)})"
                fixed = fixed[:start] + replacement + fixed[end:]
                changed = True
                break

        if not changed:
            break

    if fixes and fixed != expression:
        logger.info("Normalized operator parameters: %s -> %s", expression[:80], fixed[:80])

    return fixed, fixes
