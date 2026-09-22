import json
import time
from collections import deque
from pathlib import Path

import streamlit as st

from sudoku_solver import (
    atom,
    build_definite_kb,
    build_general_kb,
    solve_full_grid_fc,
    solve_full_grid_bc,
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


def peers_of(r, c, n, box_h, box_w):
    peers = {(r, other_c) for other_c in range(1, n + 1) if other_c != c}
    peers.update((other_r, c) for other_r in range(1, n + 1) if other_r != r)
    box_r = ((r - 1) // box_h) * box_h + 1
    box_c = ((c - 1) // box_w) * box_w + 1
    peers.update(
        (peer_r, peer_c)
        for peer_r in range(box_r, box_r + box_h)
        for peer_c in range(box_c, box_c + box_w)
        if (peer_r, peer_c) != (r, c)
    )
    return peers


def relation(source, target, box_h, box_w):
    sr, sc = source
    tr, tc = target
    if sr == tr:
        return f'row {sr}'
    if sc == tc:
        return f'column {sc}'
    box_r = (sr - 1) // box_h + 1
    box_c = (sc - 1) // box_w + 1
    return f'box ({box_r}, {box_c})'


def make_reasoning_trace(n, box_h, box_w, givens, target):
    """Create a plain-English elimination trace for the app's tutor mode."""
    candidates = {
        (r, c): set(range(1, n + 1))
        for r in range(1, n + 1)
        for c in range(1, n + 1)
    }
    queue = deque()
    steps = []
    for cell, given_value in givens.items():
        candidates[cell] = {given_value}
        queue.append((cell, given_value, True))

    processed = set()
    while queue:
        source, source_value, is_given = queue.popleft()
        if (source, source_value) in processed:
            continue
        processed.add((source, source_value))
        if is_given:
            steps.append({
                'title': f'Given: R{source[0]}C{source[1]} = {source_value}',
                'body': 'This fixed clue is an initial fact in the knowledge base.',
            })
        for peer in peers_of(*source, n, box_h, box_w):
            if source_value not in candidates[peer] or len(candidates[peer]) == 1:
                continue
            candidates[peer].remove(source_value)
            if len(candidates[peer]) == 1:
                remaining = next(iter(candidates[peer]))
                steps.append({
                    'title': f'Deduce R{peer[0]}C{peer[1]} = {remaining}',
                    'body': (
                        f'Value {source_value} is excluded because '
                        f'R{source[0]}C{source[1]} already contains it in the '
                        f'same {relation(source, peer, box_h, box_w)}. After all '
                        f'such eliminations, {remaining} is the only candidate left.'
                    ),
                })
                queue.append((peer, remaining, False))

    target_value = (
        next(iter(candidates[target])) if len(candidates[target]) == 1 else None
    )
    return steps, target_value


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
    'Test whether a proposed value is logically entailed, then open Tutor mode '
    'to inspect the deductions.'
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
        verdict = pl_bc_entails(
            kb, atom('Is', int(row), int(column), int(value))
        )
        trace, inferred_value = make_reasoning_trace(
            n, box_h, box_w, givens, (int(row), int(column))
        )
    if verdict:
        st.success(
            f'True — the knowledge base entails R{row}C{column} = {value}.'
        )
    else:
        detail = (
            f' The elimination trace instead reaches {inferred_value}.'
            if inferred_value else ''
        )
        st.error(f'False — R{row}C{column} = {value} is not entailed.{detail}')

    st.markdown('#### Tutor mode · reasoning trace')
    relevant = [
        step for step in trace
        if step['title'].startswith(f'Deduce R{int(row)}C{int(column)}')
    ]
    display_steps = trace[:12]
    if relevant and relevant[0] not in display_steps:
        display_steps.append(relevant[0])
    for number, step in enumerate(display_steps, 1):
        with st.expander(
            f'Step {number}: {step["title"]}', expanded=number <= 3
        ):
            st.write(step['body'])
    if len(trace) > len(display_steps):
        st.caption(
            f'Showing {len(display_steps)} representative steps from '
            f'{len(trace)} recorded deductions.'
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
