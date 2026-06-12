"""
Local pre-submit validation for WorldQuant FASTEXPR expressions.

This validator is intentionally conservative: it catches high-confidence
client-side mistakes before spending a WorldQuant simulation submission, while
leaving ambiguous platform-specific checks to the remote compiler.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .operator_parameter_normalizer import (
    required_argument_names,
    split_top_level_args,
)

logger = logging.getLogger(__name__)


_ARITHMETIC_OPERATORS = [
    "&&",
    "||",
    ">=",
    "<=",
    "==",
    "!=",
    "+",
    "-",
    "*",
    "/",
    "^",
    "%",
    ">",
    "<",
]

_CONSTANT_IDENTIFIERS = {
    "true",
    "false",
    "nan",
    "inf",
    "infinity",
    "gaussian",
    "cauchy",
    "uniform",
    "market",
    "sector",
    "industry",
    "subindustry",
    "country",
    "exchange",
}

_BRAIN_INACCESSIBLE_OPERATORS = {
    "generate_stats",
}

_EVENT_UNSAFE_OPERATORS = {
    "abs",
    "reverse",
    "zscore",
    "rank",
    "winsorize",
    "normalize",
    "scale",
    "log",
    "sqrt",
    "sign",
    "signed_power",
    "power",
    "subtract",
    "add",
    "multiply",
    "divide",
}


@dataclass
class LocalValidationIssue:
    """One local validation issue."""

    code: str
    message: str
    severity: str = "ERROR"
    operator_name: Optional[str] = None
    field_name: Optional[str] = None
    expected_type: Optional[str] = None
    actual_type: Optional[str] = None


@dataclass
class LocalValidationResult:
    """Result of local expression validation."""

    expression: str
    issues: List[LocalValidationIssue]

    @property
    def errors(self) -> List[LocalValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "ERROR"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary(self, max_issues: int = 3) -> str:
        if self.is_valid:
            return "Local validation passed"

        messages = [issue.message for issue in self.errors[:max_issues]]
        if len(self.errors) > max_issues:
            messages.append(f"... and {len(self.errors) - max_issues} more")
        return "; ".join(messages)


def validate_expression_locally(
    expression: str,
    operators: List[Dict],
    data_fields: List[Dict],
) -> LocalValidationResult:
    """Validate an expression against local operator and field metadata."""
    validator = LocalExpressionValidator(operators, data_fields)
    return validator.validate(expression)


class LocalExpressionValidator:
    """Lightweight recursive FASTEXPR validator used before API submission."""

    def __init__(self, operators: List[Dict], data_fields: List[Dict]):
        self.operators_by_lower = {
            str(op.get("name", "")).lower(): op
            for op in (operators or [])
            if op.get("name")
        }
        self.operator_names = {str(op.get("name", "")).lower() for op in (operators or [])}
        self.field_types = {
            str(field.get("id", "")): str(field.get("type", "REGULAR")).upper()
            for field in (data_fields or [])
            if field.get("id")
        }
        self.event_like_fields = {
            str(field.get("id", ""))
            for field in (data_fields or [])
            if field.get("id") and self._is_event_like_field(field)
        }
        self.field_ids_by_lower = {field_id.lower(): field_id for field_id in self.field_types}
        self.issues: List[LocalValidationIssue] = []

    def validate(self, expression: str) -> LocalValidationResult:
        expr = (expression or "").replace("`", "").strip()
        self.issues = []

        if not expr:
            self._add_issue("empty_expression", "Expression is empty")
            return LocalValidationResult(expr, self.issues)

        placeholders = re.findall(
            r"\b(OPERATOR\d+|operator\d+|DATA_FIELD\d+|data_field\d+)\b",
            expr,
            flags=re.IGNORECASE,
        )
        if placeholders:
            self._add_issue(
                "placeholder_remaining",
                f"Expression still contains placeholders: {', '.join(sorted(set(placeholders)))}",
            )

        paren_error = self._check_balanced_parentheses(expr)
        if paren_error:
            self._add_issue("unbalanced_parentheses", paren_error)
            return LocalValidationResult(expr, self._dedupe_issues())

        self._infer_type(expr)
        return LocalValidationResult(expr, self._dedupe_issues())

    def _infer_type(self, expression: str) -> str:
        expr = self._strip_wrapping_parentheses(expression.strip())
        if not expr:
            self._add_issue("empty_expression", "Expression contains an empty sub-expression")
            return "UNKNOWN"

        if self._is_literal(expr):
            return "LITERAL"

        split = self._split_top_level_operator(expr)
        if split:
            left, operator, right = split
            if not left or not right:
                self._add_issue(
                    "invalid_arithmetic",
                    f"Operator '{operator}' is missing an operand in '{expr[:80]}'",
                )
                return "UNKNOWN"
            self._infer_type(left)
            self._infer_type(right)
            event_field = self._event_field_in_expression(left) or self._event_field_in_expression(right)
            if event_field:
                self._add_issue(
                    "event_input_operator_mismatch",
                    f"Arithmetic operator '{operator}' does not support event/news field '{event_field}'",
                    field_name=event_field,
                )
            return "REGULAR"

        if self._has_named_argument_equals(expr):
            return "PARAMETER"

        function_call = self._parse_entire_function_call(expr)
        if function_call:
            op_name, args_text = function_call
            return self._validate_function_call(op_name, args_text, expr)

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expr):
            return self._validate_identifier(expr)

        self._add_issue("invalid_syntax", f"Could not parse expression segment: {expr[:80]}")
        return "UNKNOWN"

    def _validate_function_call(self, op_name: str, args_text: str, full_expr: str) -> str:
        op_key = op_name.lower()
        op_info = self.operators_by_lower.get(op_key)
        args = split_top_level_args(args_text)
        positional_args = [arg for arg in args if not self._has_named_argument_equals(arg)]

        if not op_info:
            self._add_issue(
                "unknown_operator",
                f"Unknown operator '{op_name}'",
                operator_name=op_name,
            )
            for arg in positional_args:
                self._infer_type(arg)
            return "UNKNOWN"

        if op_key in _BRAIN_INACCESSIBLE_OPERATORS:
            self._add_issue(
                "inaccessible_operator",
                f"Operator '{op_name}' is listed locally but is not accepted by WorldQuant Brain simulations",
                operator_name=op_name,
            )

        definition = str(op_info.get("definition", ""))
        required_names = required_argument_names(definition)
        required_count = len(required_names)

        if required_count and len(positional_args) < required_count:
            self._add_issue(
                "missing_operator_argument",
                (
                    f"Operator {op_name} expects at least {required_count} positional "
                    f"argument(s) from definition '{definition}', got {len(positional_args)}"
                ),
                operator_name=op_name,
            )

        if (
            required_count
            and len(positional_args) > required_count
            and not self._definition_is_variadic(definition)
            and not self._definition_has_optional_args(definition)
        ):
            self._add_issue(
                "too_many_operator_arguments",
                (
                    f"Operator {op_name} expects {required_count} positional "
                    f"argument(s) from definition '{definition}', got {len(positional_args)}"
                ),
                operator_name=op_name,
            )

        positional_types = [self._infer_type(arg) for arg in positional_args]

        if op_key in _EVENT_UNSAFE_OPERATORS:
            event_field = next((self._event_field_in_expression(arg) for arg in positional_args), None)
            if event_field:
                self._add_issue(
                    "event_input_operator_mismatch",
                    f"Operator {op_name} does not support event/news field '{event_field}'",
                    operator_name=op_name,
                    field_name=event_field,
                )

        if self._operator_requires_vector_input(op_key, op_info):
            vector_arg_indexes = range(min(len(positional_args), 2)) if op_key == "vector_neut" else range(min(len(positional_args), 1))
            for index in vector_arg_indexes:
                actual_type = positional_types[index]
                if actual_type not in {"VECTOR", "UNKNOWN"}:
                    field_name = self._direct_field_name(positional_args[index])
                    detail = f" field '{field_name}'" if field_name else f" expression '{positional_args[index][:60]}'"
                    self._add_issue(
                        "vector_type_mismatch",
                        f"Operator {op_name} requires VECTOR input, got {actual_type}{detail}",
                        operator_name=op_name,
                        field_name=field_name,
                        expected_type="VECTOR",
                        actual_type=actual_type,
                    )

        if op_key == "vector_neut":
            return "VECTOR"
        if op_key.startswith("vec_"):
            return "MATRIX"
        if positional_types and positional_types[0] in {"MATRIX", "VECTOR"}:
            return "MATRIX"
        return "REGULAR"

    def _validate_identifier(self, identifier: str) -> str:
        if identifier in self.field_types:
            return self.field_types[identifier]

        lower = identifier.lower()
        if lower in _CONSTANT_IDENTIFIERS:
            return "LITERAL"

        if lower in self.operator_names:
            self._add_issue(
                "operator_missing_call",
                f"Operator '{identifier}' is missing parentheses",
                operator_name=identifier,
            )
            return "UNKNOWN"

        suggested = self.field_ids_by_lower.get(lower)
        suggestion = f" (case-sensitive field id is '{suggested}')" if suggested else ""
        self._add_issue(
            "unknown_field",
            f"Unknown field '{identifier}'{suggestion}",
            field_name=identifier,
        )
        return "UNKNOWN"

    def _operator_requires_vector_input(self, op_key: str, op_info: Dict) -> bool:
        category = str(op_info.get("category", "")).lower()
        return op_key.startswith("vec_") or op_key == "vector_neut" or category == "vector"

    def _direct_field_name(self, expression: str) -> Optional[str]:
        expr = self._strip_wrapping_parentheses(expression.strip())
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", expr) and expr in self.field_types:
            return expr
        return None

    def _event_field_in_expression(self, expression: str) -> Optional[str]:
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expression):
            if token in self.event_like_fields:
                return token
        return None

    def _is_event_like_field(self, field: Dict) -> bool:
        field_type = str(field.get("type", "")).upper()
        if field_type in {"VECTOR", "EVENT"}:
            return True
        category = field.get("category", {})
        category_text = category.get("name", "") if isinstance(category, dict) else str(category)
        dataset = field.get("dataset", {})
        dataset_text = dataset.get("name", "") if isinstance(dataset, dict) else str(dataset)
        combined = " ".join(
            [
                category_text,
                dataset_text,
                str(field.get("id", "")),
                str(field.get("name", "")),
                str(field.get("description", "")),
            ]
        ).lower()
        return "news" in combined or "event" in combined

    def _parse_entire_function_call(self, expression: str) -> Optional[Tuple[str, str]]:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(", expression)
        if not match:
            return None

        open_pos = expression.find("(", match.start())
        close_pos = self._find_matching_paren(expression, open_pos)
        if close_pos == len(expression) - 1:
            return match.group(1), expression[open_pos + 1 : close_pos]
        return None

    def _split_top_level_operator(self, expression: str) -> Optional[Tuple[str, str, str]]:
        depth = 0
        for index in range(len(expression) - 1, -1, -1):
            char = expression[index]
            if char == ")":
                depth += 1
                continue
            if char == "(":
                depth -= 1
                continue
            if depth != 0:
                continue

            for operator in _ARITHMETIC_OPERATORS:
                start = index - len(operator) + 1
                if start < 0:
                    continue
                if expression[start : index + 1] != operator:
                    continue
                if operator == "-" and self._is_unary_minus(expression, start):
                    continue
                return (
                    expression[:start].strip(),
                    operator,
                    expression[index + 1 :].strip(),
                )
        return None

    def _is_unary_minus(self, expression: str, index: int) -> bool:
        if index == 0:
            return True
        previous = expression[:index].rstrip()
        return not previous or previous[-1] in "(,+-*/^%<>=!&|"

    def _strip_wrapping_parentheses(self, expression: str) -> str:
        expr = expression
        while expr.startswith("(") and expr.endswith(")"):
            close_pos = self._find_matching_paren(expr, 0)
            if close_pos != len(expr) - 1:
                break
            expr = expr[1:-1].strip()
        return expr

    def _find_matching_paren(self, expression: str, open_pos: int) -> int:
        depth = 0
        for pos in range(open_pos, len(expression)):
            if expression[pos] == "(":
                depth += 1
            elif expression[pos] == ")":
                depth -= 1
                if depth == 0:
                    return pos
        return -1

    def _check_balanced_parentheses(self, expression: str) -> Optional[str]:
        depth = 0
        for pos, char in enumerate(expression):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    return f"Unmatched closing parenthesis at character {pos}"
        if depth:
            return f"Unmatched opening parenthesis count: {depth}"
        return None

    def _has_named_argument_equals(self, expression: str) -> bool:
        depth = 0
        for pos, char in enumerate(expression):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "=" and depth == 0:
                previous_char = expression[pos - 1] if pos > 0 else ""
                next_char = expression[pos + 1] if pos + 1 < len(expression) else ""
                if previous_char in {"=", "!", "<", ">"} or next_char == "=":
                    continue
                return True
        return False

    def _is_literal(self, expression: str) -> bool:
        if re.match(r"^-?\d+(\.\d+)?$", expression):
            return True
        return expression.lower() in _CONSTANT_IDENTIFIERS

    def _definition_is_variadic(self, definition: str) -> bool:
        text = (definition or "").lower()
        return ".." in text or "..." in text or "at least" in text or "minimum" in text

    def _definition_has_optional_args(self, definition: str) -> bool:
        match = re.search(r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)", definition or "")
        if not match:
            return False
        return any("=" in arg for arg in split_top_level_args(match.group(1)))

    def _add_issue(self, code: str, message: str, **kwargs):
        self.issues.append(LocalValidationIssue(code=code, message=message, **kwargs))

    def _dedupe_issues(self) -> List[LocalValidationIssue]:
        deduped = []
        seen = set()
        for issue in self.issues:
            key = (issue.code, issue.message, issue.operator_name, issue.field_name)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return deduped
