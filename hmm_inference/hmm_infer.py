#!/usr/bin/env python3
"""Backend inference job for a hidden Markov path model with silent loops.

Input JSON schema (all ids are unique non-empty ASCII alphanumeric strings):

    {
      "states":        ["s", "f", ...],          # state id array
      "initial":       "s",                       # initial state id
      "final":         ["f", ...],                # set of terminal state ids
      "transitions": [                            # array of transitions
        {"id": "e1", "source": "s", "target": "f",
         "label": "A", "prob": "1/2"},
        {"id": "e2", "source": "s", "target": "s",
         "label": null,  "prob": "1/2"}
      ],
      "observations": ["A"]                      # array of non-empty strings
    }

Rules:
  * non-terminal states: total probability of outgoing edges <= 1, the
    remaining mass is an immediate failure; terminal states have no edges;
  * a transition whose label is null emits nothing, every other label emits
    one observation, in traversal order;
  * a successful path is finite, reaches a terminal state and emits exactly
    the observation sequence. Non-termination, failure and extra/missing
    labels are all unsuccessful;
  * path weight = product of edge probabilities.

The output contains:
  * likelihood             : sum of weights of all successful paths;
  * status                 : "impossible" when the likelihood is 0,
                             otherwise "success";
  * expected_visits        : for positive likelihood, conditional expected
                             traversal count of every transition (id sorted,
                             zero entries retained), given success;
  * best_path              : successful path of maximum weight; ties are
                             broken by fewer transitions, then by the ASCII
                             lexicographic order of the full transition id
                             sequence; includes its weight.

All numbers are reduced fractions p/q with a positive denominator.
Everything is implemented from scratch on top of fractions.Fraction; the
core summation (silent-loop elimination) and the conditional inference use
exact linear algebra, never truncated unrolling, and mass that fails or
never terminates is never attributed to successful paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Optional

ZERO = Fraction(0)
ONE = Fraction(1)


class ValidationError(Exception):
    """Raised for an illegal input; ``path`` locates the offending field."""

    def __init__(self, path: str, message: str):
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path
        self.message = message


@dataclass(frozen=True)
class Edge:
    id: str
    source: str
    target: str
    label: Optional[str]
    prob: Fraction


@dataclass
class Model:
    states: list[str]
    initial: str
    final: frozenset[str]
    edges: list[Edge]
    edge_by_id: dict[str, Edge]
    out: dict[str, list[Edge]]
    observations: list[str]


# --------------------------------------------------------------------------- #
# Validation / parsing
# --------------------------------------------------------------------------- #

def _is_nonempty_alnum(token: Any) -> bool:
    return (
        isinstance(token, str)
        and len(token) > 0
        and all(ord(ch) < 128 and ch.isalnum() for ch in token)
    )


def _parse_prob(raw: Any, path: str) -> Fraction:
    """Parse a probability written as p/q with q a positive integer."""
    if not isinstance(raw, str):
        raise ValidationError(path, "prob must be a string of the form p/q")
    parts = raw.split("/")
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        raise ValidationError(path, f"prob {raw!r} is not of the form p/q")
    p_text, q_text = parts
    if not (p_text.lstrip("-").isdigit() and q_text.isdigit()):
        raise ValidationError(path, f"prob {raw!r} must contain integers")
    try:
        value = Fraction(int(p_text), int(q_text))
    except (ZeroDivisionError, ValueError):
        raise ValidationError(path, f"prob {raw!r} has a non-positive denominator")
    if value < 0 or value > 1:
        raise ValidationError(path, f"prob {raw!r} is outside [0, 1]")
    return value


def parse_model(data: Any) -> Model:
    if not isinstance(data, dict):
        raise ValidationError("$", "input must be a JSON object")

    known = {"states", "initial", "final", "transitions", "observations"}
    unknown = set(data) - known
    if unknown:
        raise ValidationError("$", f"unknown field(s): {sorted(unknown)}")
    for key in ("states", "initial", "final", "transitions", "observations"):
        if key not in data:
            raise ValidationError(f"$.{key}", "missing required field")

    # ---- states -----------------------------------------------------------
    states_raw = data["states"]
    if not isinstance(states_raw, list) or not states_raw:
        raise ValidationError("$.states", "must be a non-empty array")
    states: list[str] = []
    seen_states: set[str] = set()
    for i, st in enumerate(states_raw):
        if not _is_nonempty_alnum(st):
            raise ValidationError(
                f"$.states[{i}]", "state id must be a unique non-empty ASCII "
                "alphanumeric string")
        if st in seen_states:
            raise ValidationError(f"$.states[{i}]", f"duplicate state id {st!r}")
        seen_states.add(st)
        states.append(st)

    # ---- initial ----------------------------------------------------------
    initial = data["initial"]
    if not _is_nonempty_alnum(initial) or initial not in seen_states:
        raise ValidationError("$.initial", f"unknown initial state {initial!r}")

    # ---- final ------------------------------------------------------------
    final_raw = data["final"]
    if not isinstance(final_raw, list):
        raise ValidationError("$.final", "must be an array of state ids")
    final_set: set[str] = set()
    for i, st in enumerate(final_raw):
        if not _is_nonempty_alnum(st) or st not in seen_states:
            raise ValidationError(f"$.final[{i}]", f"unknown terminal state {st!r}")
        if st in final_set:
            raise ValidationError(f"$.final[{i}]", f"duplicate terminal state {st!r}")
        final_set.add(st)

    # ---- observations -----------------------------------------------------
    obs_raw = data["observations"]
    if not isinstance(obs_raw, list):
        raise ValidationError("$.observations", "must be an array of strings")
    observations: list[str] = []
    for i, ob in enumerate(obs_raw):
        if not isinstance(ob, str) or ob == "":
            raise ValidationError(
                f"$.observations[{i}]", "observation must be a non-empty string")
        observations.append(ob)

    # ---- transitions ------------------------------------------------------
    trans_raw = data["transitions"]
    if not isinstance(trans_raw, list):
        raise ValidationError("$.transitions", "must be an array")
    edge_fields = {"id", "source", "target", "label", "prob"}
    edges: list[Edge] = []
    seen_edge_ids: set[str] = set()
    for i, tr in enumerate(trans_raw):
        base = f"$.transitions[{i}]"
        if not isinstance(tr, dict):
            raise ValidationError(base, "transition must be an object")
        extra = set(tr) - edge_fields
        if extra:
            raise ValidationError(base, f"unknown field(s): {sorted(extra)}")
        for key in edge_fields:
            if key not in tr:
                raise ValidationError(f"{base}.{key}", "missing required field")

        eid = tr["id"]
        if not _is_nonempty_alnum(eid):
            raise ValidationError(
                f"{base}.id",
                "transition id must be a non-empty ASCII alphanumeric string")
        if eid in seen_edge_ids:
            raise ValidationError(f"{base}.id", f"duplicate transition id {eid!r}")
        seen_edge_ids.add(eid)

        source = tr["source"]
        if not _is_nonempty_alnum(source) or source not in seen_states:
            raise ValidationError(
                f"{base}.source", f"unknown source state {source!r}")
        target = tr["target"]
        if not _is_nonempty_alnum(target) or target not in seen_states:
            raise ValidationError(
                f"{base}.target", f"unknown target state {target!r}")

        label = tr["label"]
        if label is not None and (not isinstance(label, str) or label == ""):
            raise ValidationError(
                f"{base}.label", "label must be a non-empty string or null")

        prob = _parse_prob(tr["prob"], f"{base}.prob")

        if source in final_set:
            raise ValidationError(
                f"{base}.source", f"terminal state {source!r} must have no "
                "outgoing transitions")

        edges.append(Edge(eid, source, target, label, prob))

    # outgoing probability mass: <= 1 for every non-terminal state
    out: dict[str, list[Edge]] = {st: [] for st in states}
    for edge in edges:
        out[edge.source].append(edge)
    for st in states:
        if st in final_set:
            continue
        total = sum((e.prob for e in out[st]), ZERO)
        if total > ONE:
            raise ValidationError(
                "$.transitions",
                f"outgoing probability of state {st!r} sums to {total} > 1")

    edge_by_id = {e.id: e for e in edges}
    return Model(states, initial, frozenset(final_set), edges, edge_by_id,
                 out, observations)


# --------------------------------------------------------------------------- #
# Exact linear algebra (Gauss-Jordan elimination on Fractions)
# --------------------------------------------------------------------------- #

def rref_solve(matrix: list[list[Fraction]],
               rhs_list: list[list[Fraction]]) -> list[list[Fraction]]:
    """Solve ``matrix @ x = rhs`` for every right-hand side.

    The matrix must be square and invertible; every rhs is a column vector
    of matching size. Returns one solution column per rhs.
    """
    n = len(matrix)
    if n == 0:
        return [[] for _ in rhs_list]
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be square")
    for rhs in rhs_list:
        if len(rhs) != n:
            raise ValueError("rhs size does not match matrix")

    # Augmented rows: matrix columns followed by all rhs columns.
    rows = [list(matrix[i]) + [rhs[i] for rhs in rhs_list]
            for i in range(n)]
    r_count = len(rhs_list)
    pivot_row = 0
    for col in range(n):
        pivot = next((r for r in range(pivot_row, n)
                      if rows[r][col] != ZERO), None)
        if pivot is None:
            raise ValueError("singular matrix in exact linear solve")
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row][col]
        rows[pivot_row] = [v / pivot_value for v in rows[pivot_row]]
        for r in range(n):
            if r != pivot_row and rows[r][col] != ZERO:
                factor = rows[r][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[pivot_row])]
        pivot_row += 1

    solutions = [[rows[i][n + j] for i in range(n)] for j in range(r_count)]
    return solutions


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@dataclass
class InferenceResult:
    likelihood: Fraction
    status: str
    expected_visits: dict[str, Fraction] = field(default_factory=dict)
    best_path: Optional[dict[str, Any]] = None


def _fmt(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def infer(model: Model) -> InferenceResult:
    n = len(model.observations)

    # Silent (null) transition matrix Q.
    q = {u: {v: ZERO for v in model.states} for u in model.states}
    null_edges: list[Edge] = []
    labeled_edges: list[Edge] = []
    for edge in model.edges:
        if edge.label is None:
            q[edge.source][edge.target] += edge.prob
            null_edges.append(edge)
        else:
            labeled_edges.append(edge)

    # Bad region B: states from which no labeled edge nor terminal state is
    # reachable through silent edges. A walk entering B can only keep moving
    # silently forever (or fail); it can never succeed. On B the silent row
    # sum equals 1 and I - Q is singular, so B is removed from the linear
    # systems. Non-termination mass stays there and is never distributed to
    # successful paths.
    can_exit: set[str] = set()
    changed = True
    while changed:
        changed = False
        for st in model.states:
            if st in can_exit:
                continue
            # Reaching a terminal state is a successful end of the silent
            # walk (this is what makes empty observation sequences work).
            if st in model.final:
                can_exit.add(st)
                changed = True
                continue
            # A positive-probability labeled edge is an escape out of the
            # purely-silent region (zero-probability edges carry no mass).
            if any(e.label is not None and e.prob > ZERO for e in model.out[st]):
                can_exit.add(st)
                changed = True
                continue
            if any(q[st][v] > ZERO and v in can_exit for v in model.states):
                can_exit.add(st)
                changed = True
    region = [st for st in model.states if st in can_exit]
    idx = {st: i for i, st in enumerate(region)}
    m = len(region)

    # I - Q restricted to the good region (invertible substochastic matrix).
    matrix = [[ZERO] * m for _ in range(m)]
    for i, u in enumerate(region):
        matrix[i][i] = ONE
        for j, v in enumerate(region):
            if q[u][v]:
                matrix[i][j] -= q[u][v]

    # Labeled edges grouped by the observation they must match.
    by_obs: dict[str, list[Edge]] = {}
    for edge in labeled_edges:
        by_obs.setdefault(edge.label, []).append(edge)

    # Backward pass: f_k(u) = total weight of successful suffixes that start
    # in state u on layer k (k observations still to emit).
    # Base layer: with zero observations left, success means reaching a
    # terminal state using silent edges only, so the terminal indicator is
    # closed under (I - Q)^{-1}. This also covers the empty-observation case
    # (then f_0 is exactly this solution) and silent tails after the last
    # labeled edge.
    f: list[dict[str, Fraction]] = [
        {st: ZERO for st in model.states} for _ in range(n + 1)
    ]
    base_rhs = [ONE if st in model.final else ZERO for st in region]
    (base_solved,) = rref_solve(matrix, [base_rhs])
    for i, st in enumerate(region):
        f[n][st] = base_solved[i]
    for k in range(n - 1, -1, -1):
        rhs = [ZERO] * m
        for i, u in enumerate(region):
            if u in model.final:
                # Terminal states have no outgoing edges: a path ending here
                # before all observations are emitted is unsuccessful.
                continue
            total = ZERO
            for edge in by_obs.get(model.observations[k], ()):
                if edge.source == u and edge.prob > ZERO:
                    total += edge.prob * f[k + 1][edge.target]
            rhs[i] = total
        (solved,) = rref_solve(matrix, [rhs])
        for i, st in enumerate(region):
            f[k][st] = solved[i]

    likelihood = f[0][model.initial]
    if likelihood == ZERO:
        return InferenceResult(ZERO, "impossible")

    # Forward pass, two quantities per layer k:
    #   a[k][u]  = mass arriving at u through the k-th labeled edge
    #              (a[0] is the initial state indicator, no edge yet);
    #   ap[k][v] = mass at v after the silent moves that follow that arrival,
    #              i.e. ap[k] = (I - Q)^T^{-1} a[k] restricted to the good
    #              region. Mass entering the bad region is dropped here and
    #              is never attributed to a successful path.
    # The backward f[k] already includes the silent closure *before* the next
    # labeled edge, so products ap[k] * f[k] count every path exactly once.
    transpose = [[matrix[j][i] for j in range(m)] for i in range(m)]

    def silent_forward(vector: dict[str, Fraction]) -> dict[str, Fraction]:
        rhs = [vector[st] for st in region]
        (solved,) = rref_solve(transpose, [rhs])
        return {st: (solved[i] if st in idx else ZERO)
                for i, st in enumerate(region)} | {
            st: ZERO for st in model.states if st not in idx}

    a: list[dict[str, Fraction]] = [
        {st: ZERO for st in model.states} for _ in range(n + 1)
    ]
    ap: list[dict[str, Fraction]] = [
        {st: ZERO for st in model.states} for _ in range(n + 1)
    ]
    if model.initial in idx:
        a[0][model.initial] = ONE
        ap[0] = silent_forward(a[0])
    # else: the initial state is in the bad region; likelihood would be 0.

    # Layers at which each observation label occurs (for labeled-edge counts).
    label_layers: dict[str, list[int]] = {}
    for k, ob in enumerate(model.observations):
        label_layers.setdefault(ob, []).append(k)

    for k in range(n):
        incoming: dict[str, Fraction] = {st: ZERO for st in model.states}
        for edge in by_obs.get(model.observations[k], ()):
            if edge.prob == ZERO:
                continue
            incoming[edge.target] += ap[k][edge.source] * edge.prob
        if not any(incoming[st] != ZERO for st in region):
            break  # no prefix mass reaches the next layer
        a[k + 1] = incoming
        ap[k + 1] = silent_forward(incoming)

    # Cross-check: the forward total over terminal endings must reproduce the
    # backward-computed likelihood exactly.
    forward_likelihood = sum(
        ap[n][st] * (ONE if st in model.final else ZERO) for st in region)
    if forward_likelihood != likelihood:  # pragma: no cover - internal invariant
        raise AssertionError(
            f"forward/backward likelihood mismatch: "
            f"{forward_likelihood} != {likelihood}")

    # Success-weighted traversal count of each transition = sum over all
    # successful paths of (path weight * number of traversals of the edge),
    # divided by the likelihood. This is an unconditional traversal count
    # over successful paths, not a visit-at-least-once probability.
    expected: dict[str, Fraction] = {edge.id: ZERO for edge in model.edges}
    for edge in null_edges:
        if edge.source not in idx or edge.target not in idx:
            continue
        total = ZERO
        for k in range(n + 1):
            # The source mass is taken *after* the silent closure at layer k:
            # ap[k][u] already sums the geometric series of prior silent
            # moves, so ap[k][u] * p * f[k][v] counts this edge over every
            # successful path (self-loops included).
            total += ap[k][edge.source] * edge.prob * f[k][edge.target]
        expected[edge.id] = total / likelihood
    for edge in labeled_edges:
        total = ZERO
        for k in label_layers.get(edge.label, ()):
            total += ap[k][edge.source] * edge.prob * f[k + 1][edge.target]
        expected[edge.id] = total / likelihood

    best = _best_path(model, null_edges, q, region, by_obs, f)
    return InferenceResult(likelihood, "success", expected, best)


def _best_path(model: Model, null_edges: list[Edge],
               q: dict[str, dict[str, Fraction]], region: list[str],
               by_obs: dict[str, list[Edge]],
               f: list[dict[str, Fraction]]) -> dict[str, Any]:
    """Maximum-weight successful path with the required tie breaks.

    Candidate order: (weight desc, edge count asc, id sequence ASCII asc).
    Silent parts are precomputed with a max-plus Floyd-Warshall closure over
    the silent graph; silent cycles never improve a candidate because every
    probability is <= 1 (a unit-probability cycle only adds edges).
    """
    states = model.states

    def better(cand, cur):
        if cur is None:
            return True
        if cand[0] != cur[0]:
            return cand[0] > cur[0]
        if len(cand[1]) != len(cur[1]):
            return len(cand[1]) < len(cur[1])
        return cand[1] < cur[1]

    # closure[u][v] = (weight, edge id tuple) of the best silent path u -> v.
    closure: dict[str, dict[str, Optional[tuple[Fraction, tuple[str, ...]]]]] = {
        u: {v: None for v in states} for u in states
    }
    for u in states:
        closure[u][u] = (ONE, ())
    # Parallel silent edges: keep the best one directly.
    for edge in null_edges:
        cand = (edge.prob, (edge.id,))
        if better(cand, closure[edge.source][edge.target]):
            closure[edge.source][edge.target] = cand
    for mid in states:
        for u in states:
            part1 = closure[u][mid]
            if part1 is None:
                continue
            for v in states:
                part2 = closure[mid][v]
                if part2 is None:
                    continue
                cand = (part1[0] * part2[0], part1[1] + part2[1])
                if better(cand, closure[u][v]):
                    closure[u][v] = cand

    n = len(model.observations)
    # value[k][u] = best full successful path starting at u on layer k.
    value: list[dict[str, Optional[tuple[Fraction, tuple[str, ...]]]]] = [
        {st: None for st in states} for _ in range(n + 1)
    ]
    # Base layer: with no observations left, finish at a terminal state via
    # the best silent-only path (the empty path when already terminal).
    for st in states:
        best_base = None
        for term in model.final:
            cand = closure[st][term]
            if cand is not None and better(cand, best_base):
                best_base = cand
        value[n][st] = best_base
    for k in range(n - 1, -1, -1):
        for u in states:
            if u in model.final:
                continue
            best_cand = None
            for mid in states:
                silent = closure[u][mid]
                if silent is None:
                    continue
                for edge in by_obs.get(model.observations[k], ()):
                    if edge.source != mid or edge.prob == ZERO:
                        continue
                    suffix = value[k + 1][edge.target]
                    if suffix is None:
                        continue
                    cand = (silent[0] * edge.prob * suffix[0],
                            silent[1] + (edge.id,) + suffix[1])
                    if better(cand, best_cand):
                        best_cand = cand
            value[k][u] = best_cand

    weight, edge_ids = value[0][model.initial]  # non-None: likelihood > 0
    return {
        "transition_ids": list(edge_ids),
        "weight": _fmt(weight),
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def build_output(model: Model, result: InferenceResult) -> dict[str, Any]:
    if result.status == "impossible":
        return {"likelihood": "0/1", "status": "impossible"}
    return {
        "likelihood": _fmt(result.likelihood),
        "status": "success",
        "expected_visits": [
            {"transition_id": eid, "expected_visits": _fmt(result.expected_visits[eid])}
            for eid in sorted(result.expected_visits)
        ],
        "best_path": result.best_path,
    }


def run(input_path: str, output_path: Optional[str]) -> dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    model = parse_model(data)
    result = infer(model)
    output = build_output(model, result)
    if output_path is not None:
        # The output file is replaced only after a fully successful run.
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return output


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact HMM likelihood, conditional expected traversals "
                    "and best successful path.")
    parser.add_argument("input", help="path to the input JSON file")
    parser.add_argument("-o", "--output", help="optional output JSON path")
    args = parser.parse_args(argv)
    try:
        output = run(args.input, args.output)
    except ValidationError as exc:
        json.dump({"error": exc.message, "path": exc.path},
                  sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 2
    except json.JSONDecodeError as exc:
        json.dump({"error": f"invalid JSON: {exc.msg}",
                   "path": f"${exc.pos}"}, sys.stderr, ensure_ascii=False)
        sys.stderr.write("\n")
        return 2
    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
