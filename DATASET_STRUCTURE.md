## Dataset Sample Structure

Each entry in the JSONL result files (`analyze/results/spatial_gym/*.jsonl`) contains the following keys:

### Puzzle Information
- **`id`**: Unique puzzle identifier (string)
- **`difficulty_level`**: Integer difficulty rating (1-5, where 1 is easiest and 5 is hardest)
- **`difficulty_score`**: Float difficulty score (continuous measure)
- **`grid_size`**: Object with `height` and `width` keys (puzzle dimensions)
- **`polyshapes`**: JSON string containing polyomino shape definitions
- **`puzzle_array`**: 2D array representing the puzzle grid with cell types and constraints
- **`solution_count`**: Number of valid solutions for this puzzle
- **`solutions`**: Array of solution objects, each containing `index`, `path` (list of {x, y} coordinates), and `pathLength`
- **`text_visualization`**: Human-readable YAML representation of the puzzle

### Model Results
- **`result`**: Object containing the model's attempt and analysis
  - **`puzzle_id`**: Reference to the puzzle ID
  - **`solved`**: Boolean indicating if the puzzle was correctly solved
  - **`analysis`**: Object with validation metrics:
    - `starts_at_start_ends_at_exit`: Path begins at start and ends at exit
    - `connected_line`: Path forms a connected line
    - `non_intersecting_line`: Path doesn't cross itself
    - `start_to_exit_connected`: Start and exit are connected
    - `no_rule_crossing`: No puzzle rules were violated
    - `fully_valid_path`: Overall validity (all checks passed)
  - **`processing_time`**: Time in seconds to generate the solution
  - **`extracted_path`**: List of {x, y} coordinates extracted from model output
  - **`message`**: Raw model output (typically includes thinking process and answer)
  - **`error`**: Error message if processing failed (null otherwise)

### Optional Annotations
- **`failure_annotation`**: (Optional) Manual annotation for failed attempts
  - **`completed`**: Boolean indicating if annotation is complete
  - **`failure_reasons`**: Array of failure category codes:
    - `a_planning_logical_flaw`: Logical or planning errors in approach
    - `b_misunderstood_invented_rule`: Misinterpreted or invented puzzle rules
    - `c_spatial_geometric_misjudgment`: Spatial reasoning or geometric errors
    - `d_premature_verification`: Claims solution is correct without checking key rules
    - `e_no_correction_despite_noticing`: Recognizes error but doesn't adjust the plan
    - `f_grid_coordinate_error`: Incorrect coordinates or indexing (off-by-one, swapped x/y, out of bounds)
  - **`other_reason`**: Free-text field for additional context
  - **`puzzle_id`**: Reference to the puzzle ID

- **`llm_annotation`**: (Optional) LLM judge annotation (see Annotation pipeline section)
  - **`categories`**: Array of letter codes for failure categories (e.g., ["C", "A"]):
    - `A`: Planning/logical flaw in the reasoning approach
    - `B`: Misunderstood or invented puzzle rules
    - `C`: Spatial/geometric misjudgment or miscalculation
    - `D`: Premature verification - claims correctness without checking key rules
    - `E`: No correction despite noticing - recognizes errors but doesn't adjust
    - `F`: Grid/coordinate error - off-by-one, swapped x/y, or out-of-bounds steps
  - **`explanation`**: Human-readable explanation of the failure modes
  - **`llm_raw`**: Complete raw LLM output including thinking process

### Example Sample
```json
{
  "difficulty_level": 3,
  "difficulty_score": 2.94,
  "id": "aa31d05ed8fdb273",
  "grid_size": {"height": 6, "width": 3},
  "result": {
    "puzzle_id": "aa31d05ed8fdb273",
    "solved": false,
    "analysis": {
      "fully_valid_path": false,
      "no_rule_crossing": false
    },
    "processing_time": 4.74,
    "extracted_path": [{"x": 6, "y": 0}, {"x": 6, "y": 1}]
  },
  "failure_annotation": {
    "completed": true,
    "failure_reasons": ["a_planning_logical_flaw", "c_spatial_geometric_misjudgment"],
    "other_reason": "",
    "puzzle_id": "aa31d05ed8fdb273"
  },
  "llm_annotation": {
    "categories": ["C", "A"],
    "explanation": "The model exhibits a spatial/geometric misjudgment (C) by assuming regions can fit polyshapes without verifying their actual size and layout. Additionally, the model shows a planning flaw (A) by constructing a path that appears to work step-by-step but fails to account for polyshape constraints.",
    "llm_raw": "<think>\n...\n</think>\n\n{\"categories\": [\"C\", \"A\"], \"explanation\": \"...\"}"
  }
}
```