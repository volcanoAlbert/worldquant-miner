from generation_two.core.selection_safety import filter_generation_operators


def test_concentration_prone_operators_are_excluded_from_default_generation():
    operators = [
        {"name": "rank", "category": "Cross Sectional"},
        {"name": "zscore", "category": "Cross Sectional"},
        {"name": "sign", "category": "Arithmetic"},
        {"name": "sqrt", "category": "Arithmetic"},
        {"name": "ts_mean", "category": "Time Series"},
    ]

    filtered_names = {op["name"] for op in filter_generation_operators(operators)}

    assert "rank" in filtered_names
    assert "zscore" in filtered_names
    assert "ts_mean" in filtered_names
    assert "sign" not in filtered_names
    assert "sqrt" not in filtered_names
