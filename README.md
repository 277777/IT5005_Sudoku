# IT5005 Sudoku Logic Lab

An interactive Sudoku solver built for the IT5005 Artificial Intelligence assignment. The project represents Sudoku constraints in propositional logic and demonstrates forward- and backward-chaining inference through a Streamlit interface.

## Live app

[Open Sudoku Logic Lab](https://sudokuapppy-cvr6xwcrugb56jel75jmtw.streamlit.app/)

## Features

- Select any puzzle from the supplied puzzle collection.
- Display givens and inferred values with distinct styling.
- Solve the full grid using forward or backward chaining.
- Compare inference runtimes.
- Check whether a specific cell/value proposition is entailed.
- Inspect a plain-English reasoning trace in Tutor mode.

## Logic representation

The solver provides two knowledge-base representations:

- **General propositional KB:** direct CNF clauses for at-least-one, at-most-one, row, column, box, and given constraints.
- **Definite-clause KB:** positive `Is` and `Not` symbols, elimination implications, and last-candidate rules suitable for Horn-clause inference.

The core implementation is in `sudoku_solver.py`. The Streamlit interface imports the solver rather than duplicating its inference functions.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run sudoku_app.py
```

Then open `http://localhost:8501`.

## Repository files

| File | Purpose |
|---|---|
| `sudoku_app.py` | Streamlit user interface and reasoning trace |
| `sudoku_solver.py` | Knowledge-base builders and inference algorithms |
| `puzzles.json` | Puzzle givens and reference solutions |
| `logic_.py` | Provided propositional-logic support code |
| `utils.py` | Provided utility functions |
| `requirements.txt` | Python dependencies for deployment |

## Deployment

The app is deployed with Streamlit Community Cloud from the `main` branch. Updates pushed to this repository are deployed automatically.
