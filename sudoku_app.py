import json
import time
from pathlib import Path

import streamlit as st

from sudoku_solver import (
    atom,
    build_definite_kb,
    build_general_kb,
    solve_full_grid_fc,
    solve_full_grid_bc,
    pl_fc_entails_with_trace,
    pl_bc_entails,
)


APP_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_pool():
    with (APP_DIR / 'puzzles.json').open() as file:
        raw = json.load(file)
    puzzles = []
    for puzzle in raw['puzzles']:
        givens = {
            tuple(int(part) for part in key.split('_')): value
            for key, value in puzzle['givens'].items()
        }
        puzzles.append({'givens': givens, 'given_count': puzzle['given_count']})
    return raw['n'], raw['box_h'], raw['box_w'], puzzles


def board_html(n, box_h, box_w, values, givens):
    cells = []
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            value = values.get((r, c), '')
            classes = ['cell', 'given' if (r, c) in givens else 'deduced']
            if c % box_w == 0 and c != n:
                classes.append('box-right')
            if r % box_h == 0 and r != n:
                classes.append('box-bottom')
            cells.append(f'<div class="{" ".join(classes)}">{value}</div>')
    return f'''
    <style>
      .sudoku-board {{
        display:grid; grid-template-columns:repeat({n}, minmax(34px, 52px));
        width:max-content; border:3px solid #263238; border-radius:8px;
        overflow:hidden; box-shadow:0 8px 24px rgba(0,0,0,.08);
        margin:.5rem 0 1rem;
      }}
      .cell {{
        width:52px; height:52px; display:flex; align-items:center;
        justify-content:center; font-size:1.35rem; border-right:1px solid #b0bec5;
        border-bottom:1px solid #b0bec5; box-sizing:border-box;
      }}
      .given {{background:#e8eefc; color:#163a74; font-weight:750;}}
      .deduced {{background:#fff; color:#1d6b52; font-weight:600;}}
      .box-right {{border-right:3px solid #263238;}}
      .box-bottom {{border-bottom:3px solid #263238;}}
    </style>
    <div class="sudoku-board">{"".join(cells)}</div>
    '''


def parse_atom(symbol):
    """Parse Is1_2_3 / Not1_2_3 into (prefix, row, col, value)."""
    text = str(symbol)

    for prefix in ('Is', 'Not'):
        if text.startswith(prefix):
            parts = text[len(prefix):].split('_')

            if len(parts) == 3:
                r, c, v = map(int, parts)
                return prefix, r, c, v

    return None


def symbol_text(symbol):
    """Convert an internal proposition into readable English."""
    parsed = parse_atom(symbol)

    if parsed is None:
        return str(symbol)

    prefix, r, c, v = parsed

    if prefix == 'Is':
        return f'R{r}C{c} = {v}'

    return f'R{r}C{c} ≠ {v}'


def extract_proof(search_trace, query):
    """Keep only the steps the final proof of `query` actually uses.

    pl_bc_entails records every subgoal it proves during the search,
    including ones from branches that were later abandoned.  Walking back
    from the query through the rule that proved each goal recovers the real
    proof tree.  Returned in post-order: givens first, query last.
    """
    rule_of = {}
    for premises, conclusion in search_trace:
        if premises:
            # The first rule recorded for a goal is the one that proved it;
            # its premises were all proved (and recorded) before it.
            rule_of.setdefault(conclusion, tuple(premises))

    order, used, stack = [], set(), [(query, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append((rule_of.get(node, ()), node))
            continue
        if node in used:
            continue
        used.add(node)
        stack.append((node, True))
        for premise in rule_of.get(node, ()):
            if premise not in used:
                stack.append((premise, False))
    return order


def make_reasoning_trace(kb, query, algorithm, max_steps=30):
    """Run the selected algorithm and format its actual proof trace."""

    search_trace = []

    if algorithm == 'Forward chaining':
        verdict, search_trace = pl_fc_entails_with_trace(kb, query)
    else:
        verdict = pl_bc_entails(
            kb,
            query,
            trace=search_trace
        )

    # Only a successful query has a proof to show.
    proof_trace = extract_proof(search_trace, query) if verdict else []

    facts = getattr(kb, '_bc_facts', set())
    steps = []

    for premises, conclusion in proof_trace:

        # Base fact / given
        if not premises:
            if conclusion in facts:
                steps.append({
                    'title': f'Given: {symbol_text(conclusion)}',
                    'body': (
                        f'{symbol_text(conclusion)} is an initial fact '
                        f'in the knowledge base.'
                    ),
                })
            else:
                steps.append({
                    'title': f'Already established: {symbol_text(conclusion)}',
                    'body': (
                        f'{symbol_text(conclusion)} was proved earlier '
                        f'in the {algorithm.lower()} search.'
                    ),
                })

        # Rule application
        else:
            premise_text = ', '.join(
                symbol_text(premise)
                for premise in premises
            )

            conclusion_info = parse_atom(conclusion)

            # Elimination rule:
            # Is(...) ==> Not(...)
            if (
                conclusion_info is not None
                and conclusion_info[0] == 'Not'
                and len(premises) == 1
                and parse_atom(premises[0]) is not None
                and parse_atom(premises[0])[0] == 'Is'
            ):
                steps.append({
                    'title': f'Eliminate: {symbol_text(conclusion)}',
                    'body': (
                        f'Because {symbol_text(premises[0])}, '
                        f'the Sudoku constraints imply '
                        f'{symbol_text(conclusion)}.'
                    ),
                })

            # Last-candidate rule:
            # Not(...) & Not(...) & ... ==> Is(...)
            elif (
                conclusion_info is not None
                and conclusion_info[0] == 'Is'
                and all(
                    parse_atom(premise) is not None
                    and parse_atom(premise)[0] == 'Not'
                    for premise in premises
                )
            ):
                steps.append({
                    'title': f'Deduce: {symbol_text(conclusion)}',
                    'body': (
                        f'All competing candidates have been eliminated: '
                        f'{premise_text}. Therefore '
                        f'{symbol_text(conclusion)}.'
                    ),
                })

            # Generic Horn-rule explanation
            else:
                steps.append({
                    'title': f'Deduce: {symbol_text(conclusion)}',
                    'body': (
                        f'Because {premise_text}, '
                        f'deduce {symbol_text(conclusion)}.'
                    ),
                })

    # The proof runs from the givens up to the query, so if it is too long
    # keep the final steps, which lead directly to the answer.
    if len(steps) > max_steps:
        omitted = len(steps) - max_steps
        steps = [{
            'title': f'{omitted} earlier proof steps not shown',
            'body': (
                f'The full proof has {omitted + max_steps} steps. The first '
                f'{omitted} derive supporting facts further back from the '
                f'givens; the final {max_steps} steps below lead directly '
                f'to the query.'
            ),
        }] + steps[-max_steps:]

    if verdict:
        steps.append({
            'title': 'Query proved',
            'body': (
                f'The {algorithm.lower()} proof establishes '
                f'{symbol_text(query)}.'
            ),
        })
    else:
        steps.append({
            'title': 'Query not entailed',
            'body': (
                f'{algorithm} could not establish a complete '
                f'proof for {symbol_text(query)} from the known facts.'
            ),
        })

    return verdict, steps


st.set_page_config(page_title='Sudoku Logic Lab', page_icon='🧩', layout='wide')
st.title('🧩 Sudoku Logic Lab')
st.caption('Explore how propositional rules solve a Sudoku by elimination.')

n, box_h, box_w, puzzle_pool = load_pool()
puzzle_index = st.selectbox(
    'Choose a puzzle',
    range(len(puzzle_pool)),
    format_func=lambda i: (
        f'Puzzle {i + 1} · {puzzle_pool[i]["given_count"]} givens'
    ),
)
puzzle = puzzle_pool[puzzle_index]
givens = puzzle['givens']

if st.session_state.get('puzzle_index') != puzzle_index:
    st.session_state.puzzle_index = puzzle_index
    st.session_state.pop('solved_grid', None)
    st.session_state.pop('solve_time', None)
    st.session_state.pop('solve_algorithm', None)
    # A trace belongs to one puzzle's KB.  Clear it only when that KB changes,
    # not when Streamlit reruns because the solver radio button changes.
    st.session_state.pop('entailment_result', None)

left, right = st.columns([1.2, 1], gap='large')
with left:
    st.subheader('Puzzle board')
    board_values = st.session_state.get('solved_grid', givens)
    st.markdown(
        board_html(n, box_h, box_w, board_values, givens),
        unsafe_allow_html=True,
    )
    st.caption('Blue cells are givens; green cells are inferred by the solver.')

with right:
    st.subheader('Full-grid solver')
    algorithm = st.radio(
        'Inference algorithm',
        ['Forward chaining', 'Backward chaining'],
        horizontal=True,
    )
    if st.button('Solve the full grid', type='primary', use_container_width=True):
        solver = (
            solve_full_grid_fc
            if algorithm == 'Forward chaining'
            else solve_full_grid_bc
        )
        started = time.perf_counter()
        solved = solver(n, box_h, box_w, givens)
        elapsed = time.perf_counter() - started
        st.session_state.solved_grid = solved
        st.session_state.solve_time = elapsed
        st.session_state.solve_algorithm = algorithm
        st.rerun()

    if 'solve_time' in st.session_state:
        if len(st.session_state.solved_grid) == n * n:
            st.success(
                f'{st.session_state.solve_algorithm} solved all {n * n} cells '
                f'in {st.session_state.solve_time:.3f} seconds.'
            )
        else:
            st.warning(
                f'The rules inferred {len(st.session_state.solved_grid)} of '
                f'{n * n} cells in {st.session_state.solve_time:.3f} seconds.'
            )

st.divider()
st.subheader('Ask the knowledge base')
st.write(
    'Test whether a proposed value is logically entailed. Tutor mode shows the '
    'actual proof produced by the selected inference algorithm.'
)

q1, q2, q3 = st.columns(3)
with q1:
    row = st.number_input('Row', min_value=1, max_value=n, value=1, step=1)
with q2:
    column = st.number_input('Column', min_value=1, max_value=n, value=1, step=1)
with q3:
    value = st.number_input('Value', min_value=1, max_value=n, value=1, step=1)

if st.button('Check entailment', use_container_width=True):
    with st.spinner('Following the rules that can support this query…'):
        kb = build_definite_kb(n, box_h, box_w, givens)

        query = atom(
            'Is',
            int(row),
            int(column),
            int(value)
        )

        verdict, trace = make_reasoning_trace(
            kb,
            query,
            algorithm
        )

    # Widget changes trigger a Streamlit rerun.  Persist the completed query
    # so its verdict and trace do not disappear after choosing FC or BC above.
    st.session_state.entailment_result = {
        'row': int(row),
        'column': int(column),
        'value': int(value),
        'algorithm': algorithm,
        'verdict': verdict,
        'trace': trace,
    }

result = st.session_state.get('entailment_result')

# Switching the solver radio button is itself a Streamlit rerun. Regenerate
# the stored query with the newly selected algorithm so Tutor mode always
# matches the visible selection.
if result is not None and result['algorithm'] != algorithm:
    kb = build_definite_kb(n, box_h, box_w, givens)
    query = atom(
        'Is',
        result['row'],
        result['column'],
        result['value']
    )
    verdict, trace = make_reasoning_trace(kb, query, algorithm)
    result = {
        **result,
        'algorithm': algorithm,
        'verdict': verdict,
        'trace': trace,
    }
    st.session_state.entailment_result = result

if result is not None:
    if result['verdict']:
        st.success(
            'True — the knowledge base entails '
            f"R{result['row']}C{result['column']} = {result['value']}."
        )
    else:
        st.error(
            'False — '
            f"R{result['row']}C{result['column']} = {result['value']} "
            'is not entailed.'
        )

    st.markdown(f'#### Tutor mode · {result["algorithm"].lower()} proof')

    for number, step in enumerate(result['trace'], 1):
        with st.expander(
            f'Step {number}: {step["title"]}',
            expanded=(number > len(result['trace']) - 3)
        ):
            st.write(step['body'])

with st.expander('About the two knowledge bases'):
    st.write(
        'The general knowledge base stores the direct CNF Sudoku constraints. '
        'The definite knowledge base uses positive Is/Not symbols, elimination '
        'rules, and last-candidate rules so Horn-clause inference can be used.'
    )
    st.caption(
        'The app imports both builders from sudoku_solver.py; solver logic is '
        'not duplicated in this interface.'
    )
