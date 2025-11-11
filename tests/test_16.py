import pytest
from definition_1835d313033345b591bb30fafb6ad5c5 import validate_latex_formatting_in_markdown_cells


def _assert_valid_issue_record(record, expected_cell_index=None):
    # Support either dict-based or tuple/list-based issue records
    if isinstance(record, dict):
        assert 'cell_index' in record and 'position' in record and 'message' in record
        if expected_cell_index is not None:
            assert record['cell_index'] == expected_cell_index
        assert isinstance(record['position'], int) and record['position'] >= 0
        assert isinstance(record['message'], str) and record['message'].strip()
    else:
        assert isinstance(record, (list, tuple)) and len(record) >= 3
        if expected_cell_index is not None:
            assert record[0] == expected_cell_index
        assert isinstance(record[1], int) and record[1] >= 0
        assert isinstance(record[2], str) and record[2].strip()


def test_no_issues_returns_empty_list():
    cells = [
        "This notebook explains inline math like $a + b = c$ clearly.",
        "And shows display equations:\n\n$$E = mc^2$$\n\nwith proper formatting."
    ]
    result = validate_latex_formatting_in_markdown_cells(cells)
    assert isinstance(result, list)
    assert result == []


@pytest.mark.parametrize("cells, description", [
    (["This is broken inline math: $a + b"], "unmatched inline $"),
    (["Equation block missing end:\n\n$$a^2 + b^2 = c^2"], "unmatched display $$"),
    (["Mismatched delimiters: $a + b$$"], "mismatched $ and $$"),
])
def test_detects_latex_issues(cells, description):
    issues = validate_latex_formatting_in_markdown_cells(cells)
    assert isinstance(issues, list) and len(issues) >= 1
    _assert_valid_issue_record(issues[0], expected_cell_index=0)
    msg = issues[0]['message'] if isinstance(issues[0], dict) else issues[0][2]
    assert any(k in msg.lower() for k in ["unmatched", "mismatch", "delimiter", "incorrect"])


def test_invalid_input_type_raises_typeerror():
    with pytest.raises((TypeError, ValueError)):
        validate_latex_formatting_in_markdown_cells("not a list")