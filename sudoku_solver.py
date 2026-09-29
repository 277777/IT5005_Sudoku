"""IT5005 Assignment 1: student implementation file.

Implement the functions marked below. Do not modify utils.py or logic_.py.
"""

from utils import *
from logic_ import (
    PropKB,
    PropDefiniteKB,
    associate,
    expr,
    parse_definite_clause,
    pl_fc_entails,
)


class _IndexedPropDefiniteKB(PropDefiniteKB):
    """PropDefiniteKB with the premise lookup cached for practical runtimes."""

    def __init__(self):
        super().__init__()
        self._premise_index = {}
        self._capture_inferred = False
        self._inferred_symbols = set()

    def tell(self, sentence):
        super().tell(sentence)
        if sentence.op == '==>':
            premises, _ = parse_definite_clause(sentence)
            for premise in premises:
                self._premise_index.setdefault(premise, []).append(sentence)

    def clauses_with_premise(self, premise):
        if self._capture_inferred:
            self._inferred_symbols.add(premise)
        return self._premise_index.get(premise, [])


# Do not change this function; it is used to create atomic propositions.
def atom(prefix, r, c, v):
    """prefix is 'Is' or 'Not'. Returns the Expr for e.g. Is3_2_4."""
    return expr(f'{prefix}{r}_{c}_{v}')


def build_general_kb(n, box_h, box_w, givens):
    """Return a PropKB encoding this n x n Sudoku's constraints plus the given
    cells, as general clauses.

    Parameters
    ----------
    n, box_h, box_w : int
    givens : dict[(int, int), int]

    Returns
    -------
    PropKB
    """
    kb = PropKB()

    # Every cell contains at least one value.
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            kb.tell(associate('|', [atom('Is', r, c, v)
                                    for v in range(1, n + 1)]))

    # Every cell contains at most one value.
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v1 in range(1, n + 1):
                for v2 in range(v1 + 1, n + 1):
                    kb.tell(~atom('Is', r, c, v1) |
                            ~atom('Is', r, c, v2))

    # A value occurs at most once in each row and column.
    for v in range(1, n + 1):
        for r in range(1, n + 1):
            for c1 in range(1, n + 1):
                for c2 in range(c1 + 1, n + 1):
                    kb.tell(~atom('Is', r, c1, v) |
                            ~atom('Is', r, c2, v))
        for c in range(1, n + 1):
            for r1 in range(1, n + 1):
                for r2 in range(r1 + 1, n + 1):
                    kb.tell(~atom('Is', r1, c, v) |
                            ~atom('Is', r2, c, v))

    # A value occurs at most once in each box.
    for box_r in range(1, n + 1, box_h):
        for box_c in range(1, n + 1, box_w):
            cells = [(r, c)
                     for r in range(box_r, box_r + box_h)
                     for c in range(box_c, box_c + box_w)]
            for v in range(1, n + 1):
                for i in range(len(cells)):
                    for j in range(i + 1, len(cells)):
                        r1, c1 = cells[i]
                        r2, c2 = cells[j]
                        kb.tell(~atom('Is', r1, c1, v) |
                                ~atom('Is', r2, c2, v))

    # Fixed clues are unit clauses.
    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))

    return kb


def build_definite_kb(n, box_h, box_w, givens):
    """Return a PropDefiniteKB encoding this n x n Sudoku's constraints plus
    the given cells, using elimination + last-candidate reasoning.

    Parameters
    ----------
    n, box_h, box_w : int
    givens : dict[(int, int), int] -- {(row, col): value}, 1-indexed

    Returns
    -------
    PropDefiniteKB
    """
    kb = _IndexedPropDefiniteKB()

    # Given cells are the initial facts.
    for (r, c), v in givens.items():
        kb.tell(atom('Is', r, c, v))

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                is_value = atom('Is', r, c, v)

                # At most one value in a cell.
                for other_v in range(1, n + 1):
                    if other_v != v:
                        kb.tell(is_value |'==>'|
                                atom('Not', r, c, other_v))

                # Row and column eliminations.
                for other_c in range(1, n + 1):
                    if other_c != c:
                        kb.tell(is_value |'==>'|
                                atom('Not', r, other_c, v))
                for other_r in range(1, n + 1):
                    if other_r != r:
                        kb.tell(is_value |'==>'|
                                atom('Not', other_r, c, v))

                # Box elimination.  Peers already covered by the row or
                # column rules are skipped to avoid duplicate clauses.
                box_r = ((r - 1) // box_h) * box_h + 1
                box_c = ((c - 1) // box_w) * box_w + 1
                for peer_r in range(box_r, box_r + box_h):
                    for peer_c in range(box_c, box_c + box_w):
                        if peer_r != r and peer_c != c:
                            kb.tell(is_value |'==>'|
                                    atom('Not', peer_r, peer_c, v))

            # If all other candidates are eliminated, the remaining one is
            # the cell's value.  This replaces the non-Horn at-least-one
            # disjunction with n definite last-candidate implications.
            for v in range(1, n + 1):
                eliminated = [atom('Not', r, c, other_v)
                              for other_v in range(1, n + 1)
                              if other_v != v]
                kb.tell(associate('&', eliminated) |'==>'|
                        atom('Is', r, c, v))

    return kb


def solve_full_grid_fc(n, box_h, box_w, givens):
    """Solve the whole puzzle using build_definite_kb + pl_fc_entails.

    Returns
    -------
    dict[(int, int), int] -- {(row, col): value} for every cell
    """
    kb = build_definite_kb(n, box_h, box_w, givens)
    # Exhaust the supplied forward-chaining procedure once and capture every
    # proposition it processes.  Asking an intentionally absent symbol makes
    # pl_fc_entails run to closure instead of returning at the first match.
    kb._capture_inferred = True
    pl_fc_entails(kb, expr('ForwardClosureSentinel'))
    kb._capture_inferred = False
    entailed = kb._inferred_symbols

    solved = {}
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                if atom('Is', r, c, v) in entailed:
                    solved[(r, c)] = v
                    break
    return solved


def pl_fc_entails_with_trace(kb, query):
    """Run forward chaining and return its successful rule-firing trace.

    This mirrors the supplied ``pl_fc_entails`` agenda/count algorithm while
    recording each rule at the moment all of its premises become known.
    """
    count = {}
    agenda = []

    for clause in kb.clauses:
        if clause.op == '==>':
            premises, _ = parse_definite_clause(clause)
            count[clause] = len(premises)
        else:
            agenda.append(clause)

    inferred = {}
    trace = []

    while agenda:
        proposition = agenda.pop()

        if proposition == query:
            return True, trace

        if inferred.get(proposition, False):
            continue

        inferred[proposition] = True

        for clause in kb.clauses_with_premise(proposition):
            count[clause] -= 1

            if count[clause] == 0:
                premises, conclusion = parse_definite_clause(clause)
                trace.append((tuple(premises), conclusion))
                agenda.append(conclusion)

    return False, trace


def pl_bc_entails(kb, query, trace=None):
    """Return True iff query is entailed by a propositional definite KB.

    Parameters
    ----------
    kb : PropDefiniteKB
    query : Expr
    trace : list or None
        Optional list used to record successful proof steps as
        (premises, conclusion) tuples.

    Returns
    -------
    bool
    """

    # Build reusable fact/rule indexes once.
    if not hasattr(kb, '_bc_rules_by_conclusion'):
        kb._bc_rules_by_conclusion = {}
        kb._bc_facts = {
            clause for clause in kb.clauses
            if clause.op != '==>'
        }
        kb._bc_proved = set(kb._bc_facts)

        for clause in kb.clauses:
            if clause.op == '==>':
                premises, conclusion = parse_definite_clause(clause)
                kb._bc_rules_by_conclusion.setdefault(
                    conclusion, []
                ).append(tuple(premises))

    proved = kb._bc_proved

    # Prevent duplicate entries in the optional reasoning trace.
    seen_trace = set()

    while True:
        # Failure is only temporary within one pass.
        # A later pass may succeed after new propositions are proved.
        failed = set()
        before = len(proved)

        def prove(goal, visiting):
            # Base case: already known/proved.
            if goal in proved:
                if trace is not None:
                    step = ((), goal)
                    if step not in seen_trace:
                        trace.append(step)
                        seen_trace.add(step)
                return True

            # This goal already failed during the current pass.
            if goal in failed:
                return False

            # Cycle detected on the current recursive proof path.
            if goal in visiting:
                return False

            visiting.add(goal)

            # OR over all rules whose conclusion matches the goal.
            for premises in kb._bc_rules_by_conclusion.get(goal, []):

                # AND over all premises of one rule.
                if all(prove(premise, visiting) for premise in premises):
                    visiting.remove(goal)
                    proved.add(goal)

                    if trace is not None:
                        step = (tuple(premises), goal)
                        if step not in seen_trace:
                            trace.append(step)
                            seen_trace.add(step)

                    return True

            visiting.remove(goal)
            failed.add(goal)
            return False

        # Try to prove the requested query.
        if prove(query, set()):
            return True

        # If this pass proved nothing new, another pass cannot make progress.
        if len(proved) == before:
            return False


def solve_full_grid_bc(n, box_h, box_w, givens):
    """Solve the whole puzzle using build_definite_kb + your own pl_bc_entails.

    For each cell, try each candidate value until pl_bc_entails confirms one
    -- the same per-cell strategy as solve_full_grid_fc, but backed by
    backward chaining instead of a single shared forward-chaining pass.

    Returns
    -------
    dict[(int, int), int] -- {(row, col): value} for every cell
    """
    kb = build_definite_kb(n, box_h, box_w, givens)
    solved = {}
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            for v in range(1, n + 1):
                if pl_bc_entails(kb, atom('Is', r, c, v)):
                    solved[(r, c)] = v
                    break
    return solved
