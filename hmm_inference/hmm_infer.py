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

    # Group visible edges for the existing best-path calculation.
    by_obs: dict[str, list[Edge]] = {}
    for edge in labeled_edges:
        by_obs.setdefault(edge.label, []).append(edge)

    # Enumerate successful branches in the observation-expanded graph.
    # Each branch carries its own visited nodes and traversal counts.
    likelihood = ZERO
    weighted_counts = {edge.id: ZERO for edge in model.edges}
    start = (model.initial, 0)
    pending = [(model.initial, 0, ONE, frozenset({start}), {})]
    while pending:
        state, consumed, weight, visited, counts = pending.pop()
        if state in model.final:
            if consumed == n:
                likelihood += weight
                for eid, count in counts.items():
                    weighted_counts[eid] += weight * count
            continue
        for edge in model.out[state]:
            if edge.prob == ZERO:
                continue
            next_consumed = consumed
            if edge.label is not None:
                if consumed == n or edge.label != model.observations[consumed]:
                    continue
                next_consumed += 1
            node = (edge.target, next_consumed)
            if node in visited:
                continue
            next_counts = counts.copy()
            next_counts[edge.id] = next_counts.get(edge.id, 0) + 1
            pending.append((edge.target, next_consumed, weight * edge.prob,
                            visited | {node}, next_counts))

    if likelihood == ZERO:
        return InferenceResult(ZERO, "impossible")
    expected = {eid: total / likelihood
                for eid, total in weighted_counts.items()}
    # The best-path helper computes its own closure; it does not use f.
    best = _best_path(model, null_edges, q, model.states, by_obs, [])
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
