"""
Safety filters for template-generation operator and field selection.

These filters intentionally favor stable, ordinary FASTEXPR expressions. News
and vector/event fields can be useful, but they require vector/reduce handling;
randomly mixing them with arithmetic operators produces many compiler FAILs.
"""

from typing import Dict, List


UNSAFE_GENERATION_OPERATORS = {
    "generate_stats",
    # These often create coarse or compressed signals that look strong in raw
    # metrics but fail WorldQuant's concentration/sub-universe checks.
    "sign",
    "sqrt",
}

STABLE_GENERATION_CATEGORIES = {
    "Arithmetic",
    "Time Series",
    "Cross Sectional",
}


def is_event_or_vector_field(field: Dict) -> bool:
    """Return True for fields that usually need vector/event-specific handling."""
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


def filter_generation_fields(fields: List[Dict]) -> List[Dict]:
    """Prefer non-vector, non-news fields for generic random generation."""
    safe_fields = [field for field in (fields or []) if not is_event_or_vector_field(field)]
    return safe_fields or (fields or [])


def filter_generation_operators(operators: List[Dict]) -> List[Dict]:
    """Prefer operators that are broadly usable in ordinary alpha expressions."""
    safe_operators = []
    for operator in operators or []:
        name = str(operator.get("name", "")).lower()
        category = str(operator.get("category", ""))
        if not name or name in UNSAFE_GENERATION_OPERATORS:
            continue
        if category not in STABLE_GENERATION_CATEGORIES:
            continue
        safe_operators.append(operator)
    return safe_operators or (operators or [])
