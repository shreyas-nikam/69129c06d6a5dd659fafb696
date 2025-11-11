import pytest
from definition_cedd8809e31f4690bba1d29a23d380ba import validate_latex_formatting_in_markdown_cells

@pytest.mark.parametrize("input_markdown_cells, expected_output", [
    # Test Case 1: All valid LaTeX formatting and no issues (includes cells with both inline and display math,
    #              plain text cells, and an empty cell).
    (["This cell contains $inline$ math and also $$display$$ math.", 
      "Another cell with just plain text.", 
      "$$x^2 + y^2 = z^2$$ is a valid display equation. And $E=mc^2$ is inline.", 
      ""], 
     []),

    # Test Case 2: Unmatched inline LaTeX delimiter.
    (["An opening $ without a closing one in this cell.", 
      "Another cell is fine $a^2 + b^2 = c^2$ but this one isn't $x"],
     [{"cell_index": 0, "position": 11, "message": "Unmatched inline LaTeX delimiter '$' found in cell 0 at position 11."},
      {"cell_index": 1, "position": 61, "message": "Unmatched inline LaTeX delimiter '$' found in cell 1 at position 61."}]),

    # Test Case 3: Unmatched display LaTeX delimiter.
    (["A cell with $$display math that is unmatched.", 
      "This cell has $$a^2$$ but also another one that's $$unmatched."],
     [{"cell_index": 0, "position": 14, "message": "Unmatched display LaTeX delimiter '$$' found in cell 0 at position 14."},
      {"cell_index": 1, "position": 47, "message": "Unmatched display LaTeX delimiter '$$' found in cell 1 at position 47."}]),

    # Test Case 4: Multiple types of issues (inline and display) across different cells.
    (["Cell 0: $a+b$ and $c+d", 
      "Cell 1: Valid equation $$x^2+y^2$$ but this one is $$unmatched", 
      "Cell 2: $e=f$ and $$g-h$$"],
     [{"cell_index": 0, "position": 20, "message": "Unmatched inline LaTeX delimiter '$' found in cell 0 at position 20."},
      {"cell_index": 1, "position": 39, "message": "Unmatched display LaTeX delimiter '$$' found in cell 1 at position 39."}]),

    # Test Case 5: Invalid input types (non-list input).
    # This covers the edge case where the input is not a list of strings as expected.
    ("a simple string, not a list of cells", TypeError),
])
def test_validate_latex_formatting_in_markdown_cells(input_markdown_cells, expected_output):
    try:
        actual_issues = validate_latex_formatting_in_markdown_cells(input_markdown_cells)
        # Sort lists of dictionaries for consistent comparison, as the order of issues might vary
        # depending on the function's internal implementation.
        if isinstance(actual_issues, list) and all(isinstance(item, dict) for item in actual_issues):
            actual_issues.sort(key=lambda x: (x.get("cell_index", -1), x.get("position", -1), x.get("message", "")))
            expected_output.sort(key=lambda x: (x.get("cell_index", -1), x.get("position", -1), x.get("message", "")))
        assert actual_issues == expected_output
    except Exception as e:
        # For expected exceptions (like TypeError), assert the type of the raised exception.
        assert isinstance(e, expected_output)

