import json
import html
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


def board_html(n, box_h, box_w, values, givens, highlight=None):
    cells = []
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            value = values.get((r, c), '')
            classes = ['cell', 'given' if (r, c) in givens else 'deduced']
            if (r, c) == highlight:
                classes.append('highlight')
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
      .highlight {{background:#fff3bf; box-shadow:inset 0 0 0 4px #f59f00;}}
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


def elimination_reason(source, target, box_h, box_w):
    """Explain why a solved source cell eliminates a target candidate."""
    _, sr, sc, sv = parse_atom(source)
    _, tr, tc, _ = parse_atom(target)

    if (sr, sc) == (tr, tc):
        return f'R{tr}C{tc} already has value {sv}'
    if sr == tr:
        return f'row {tr} already contains {sv} at C{sc}'
    if sc == tc:
        return f'column {tc} already contains {sv} at R{sr}'

    box_r = (tr - 1) // box_h + 1
    box_c = (tc - 1) // box_w + 1
    return f'box ({box_r}, {box_c}) already contains {sv} at R{sr}C{sc}'


def make_forward_solving_trace(kb, box_h, box_w):
    """Capture the chronological cell deductions of a full FC closure."""
    # This proposition does not occur in the Sudoku KB, so the instrumented
    # algorithm runs until its agenda is empty and records every fired rule.
    _, search_trace = pl_fc_entails_with_trace(
        kb,
        atom('ForwardTraceComplete', 0, 0, 0),
    )
    facts = {
        clause for clause in kb.clauses
        if clause.op != '==>'
    }
    known_values = {}
    for fact in facts:
        fact_info = parse_atom(fact)
        if fact_info is not None and fact_info[0] == 'Is':
            _, fact_r, fact_c, fact_v = fact_info
            known_values[(fact_r, fact_c)] = fact_v
    steps = []

    for premises, conclusion in search_trace:
        conclusion_info = parse_atom(conclusion)

        # Show only new solved-cell values. Candidate eliminations remain
        # grouped inside the deduction that consumes them.
        if conclusion_info is None or conclusion_info[0] != 'Is':
            continue

        _, r, c, v = conclusion_info
        if (r, c) in known_values:
            continue

        reasons = []
        for eliminated in premises:
            eliminated_info = parse_atom(eliminated)
            if eliminated_info is not None:
                candidate = eliminated_info[3]
                source = None
                for (source_r, source_c), source_v in known_values.items():
                    same_box = (
                        (source_r - 1) // box_h == (r - 1) // box_h
                        and (source_c - 1) // box_w == (c - 1) // box_w
                    )
                    if (
                        source_v == candidate
                        and (source_r, source_c) != (r, c)
                        and (source_r == r or source_c == c or same_box)
                    ):
                        source = atom('Is', source_r, source_c, source_v)
                        break

                if source is not None:
                    reason = elimination_reason(
                        source, eliminated, box_h, box_w
                    )
                else:
                    reason = 'an earlier forward-chaining rule eliminated it'
                reasons.append(f'{candidate} is removed because {reason}')

        explanation = '; '.join(reasons)
        if not explanation:
            explanation = 'all competing candidates have been eliminated'

        steps.append({
            'title': f'Deduce: R{r}C{c} = {v}',
            'body': (
                f'For R{r}C{c}, {explanation}. '
                f'Therefore {v} is the only remaining value.'
            ),
            'cell': (r, c),
            'value': v,
            'kind': 'deduced',
        })
        known_values[(r, c)] = v

    return steps


def trace_board_state(givens, steps, position):
    """Build the visual board after the selected reasoning step."""
    values = dict(givens)
    for step in steps[:position]:
        values[step['cell']] = step['value']
    highlight = steps[position - 1]['cell'] if position else None
    return values, highlight


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
    st.session_state.pop('fc_reasoning_trace', None)

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
st.subheader('Check entailment')
st.write(
    'Test whether one proposed cell value follows from the knowledge base '
    'using the inference algorithm selected above.'
)

q1, q2, q3 = st.columns(3)
with q1:
    row = st.number_input('Row', min_value=1, max_value=n, value=1, step=1)
with q2:
    column = st.number_input('Column', min_value=1, max_value=n, value=1, step=1)
with q3:
    value = st.number_input('Value', min_value=1, max_value=n, value=1, step=1)

check_clicked = st.button('Check entailment', use_container_width=True)

if check_clicked:
    with st.spinner('Following the rules that can support this query…'):
        kb = build_definite_kb(n, box_h, box_w, givens)

        query = atom(
            'Is',
            int(row),
            int(column),
            int(value)
        )

        if algorithm == 'Forward chaining':
            verdict, _ = pl_fc_entails_with_trace(kb, query)
        else:
            verdict = pl_bc_entails(kb, query)

    # Widget changes trigger a Streamlit rerun.  Persist the completed query
    # so its verdict and trace do not disappear after choosing FC or BC above.
    st.session_state.entailment_result = {
        'row': int(row),
        'column': int(column),
        'value': int(value),
        'algorithm': algorithm,
        'verdict': verdict,
    }

result = st.session_state.get('entailment_result')

if result is not None:
    if result['verdict']:
        st.success(
            f'{result["algorithm"]}: True — the knowledge base entails '
            f"R{result['row']}C{result['column']} = {result['value']}."
        )
    else:
        st.error(
            f'{result["algorithm"]}: False — '
            f"R{result['row']}C{result['column']} = {result['value']} "
            'is not entailed.'
        )

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

st.divider()
st.subheader('Forward-chaining reasoning trace')
st.write(
    'Watch forward chaining solve the whole puzzle. Each card is one new '
    'cell value inferred during the run and is independent of entailment '
    'checking.'
)

if st.button(
    'Generate forward-chaining reasoning trace',
    use_container_width=True,
    type='primary',
):
    with st.spinner('Running forward chaining and recording deductions…'):
        trace_kb = build_definite_kb(n, box_h, box_w, givens)
        st.session_state.fc_reasoning_trace = make_forward_solving_trace(
            trace_kb,
            box_h,
            box_w,
        )

fc_trace = st.session_state.get('fc_reasoning_trace')
if fc_trace is not None:
    if fc_trace:
        with st.expander(
            f'Forward-chaining reasoning trace ({len(fc_trace)} deductions)',
            expanded=True,
        ):
            st.success(
                f'Forward chaining inferred {len(fc_trace)} new cell values. '
                'Open any step to inspect that point in the solve.'
            )
            st.caption(
                'Blue numbers are original clues, green numbers were inferred '
                'in earlier steps, and the newest deduction is highlighted in '
                'yellow.'
            )

            st.markdown(
                '''
                <style>
                  details.trace-step {
                    border:1px solid rgba(49,51,63,.18); border-radius:8px;
                    margin:.45rem 0; padding:.55rem .75rem;
                  }
                  details.trace-step > summary {
                    cursor:pointer; font-weight:650;
                  }
                  details.trace-step[open] > summary {margin-bottom:.6rem;}
                </style>
                ''',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<details class="trace-step">'
                f'<summary>🟦 Starting puzzle ({len(givens)} clues)</summary>'
                f'{board_html(n, box_h, box_w, givens, givens)}'
                '</details>',
                unsafe_allow_html=True,
            )

            for number, step in enumerate(fc_trace, 1):
                board_values, highlighted = trace_board_state(
                    givens,
                    fc_trace,
                    number,
                )
                st.markdown(
                    '<details class="trace-step">'
                    f'<summary>🟨 Step {number}: '
                    f'{html.escape(step["title"])}</summary>'
                    f'{board_html(n, box_h, box_w, board_values, givens, highlight=highlighted)}'
                    f'<p>{html.escape(step["body"])}</p>'
                    '</details>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning('Forward chaining did not infer any new cell values.')
