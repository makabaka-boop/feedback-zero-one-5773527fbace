#!/usr/bin/env python3
"""Cross-validation by brute-force path enumeration (randomized models).

The brute-force engine is deliberately independent of the linear algebra in
hmm_infer: it expands every concrete walk up to a depth bound.

Two modes:
  * models whose silent subgraph is a DAG have finitely many walks (labeled
    edges are bounded by the observation count), so enumeration is exact and
    likelihood, conditional expectations and the best path are compared with
    Fraction equality;
  * models containing silent cycles are enumerated only up to a depth bound.
    Repeating a (state, emitted) pair is NOT pruned: such a walk may still
    escape and succeed. Instead the total weight of the cut-off frontier is
    tracked, giving the rigorous interval
        bf_likelihood  <=  exact likelihood  <=  bf_likelihood + frontier.
    The best path is still compared exactly: an optimal walk never repeats a
    (state, emitted) pair, since deleting a silent cycle yields a shorter
    valid walk of equal-or-greater weight, so its length is bounded by
    (n + 1) * |states|.

Run:  python3 -I test_bruteforce_crosscheck.py
"""

from __future__ import annotations

import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_infer import infer, parse_model  # noqa: E402

MAX_STEPS = 60
TRIALS = 1200


def brute_force(model):
    """Enumerate walks up to MAX_STEPS with memoization.

    Returns (likelihood, visits, best, frontier_weight, hit_bound) where
    path weights are relative to the initial state (entry weight 1).
    Memoization key: (state, emitted count, remaining depth).
    """
    out = model.out
    n = len(model.observations)
    memo = {}

    def better_best(cand, cur):
        if cur is None:
            return True
        if cand[0] != cur[0]:
            return cand[0] > cur[0]
        if len(cand[1]) != len(cur[1]):
            return len(cand[1]) < len(cur[1])
        return cand[1] < cur[1]

    def solve(state, emitted, depth):
        key = (state, emitted, depth)
        if key in memo:
            return memo[key]
        likelihood = Fraction(0)
        visits = {e.id: Fraction(0) for e in model.edges}
        frontier = Fraction(0)
        hit = False
        best = None

        if emitted == n and state in model.final:
            memo[key] = (Fraction(1), visits, (Fraction(1), ()),
                         Fraction(0), False)
            return memo[key]
        if depth == 0:
            # Every continuation together carries at most this prefix mass.
            memo[key] = (Fraction(0), visits, None, Fraction(1), True)
            return memo[key]

        for edge in out[state]:
            if edge.prob == 0:
                continue
            if edge.label is None:
                new_emitted = emitted
            else:
                if emitted >= n or edge.label != model.observations[emitted]:
                    continue  # wrong or extra label: unsuccessful
                new_emitted = emitted + 1
            sub_l, sub_v, sub_best, sub_front, sub_hit = solve(
                edge.target, new_emitted, depth - 1)
            p = edge.prob
            likelihood += p * sub_l
            for eid, value in sub_v.items():
                visits[eid] += p * value
            visits[edge.id] += p * sub_l  # this edge, once per suffix path
            frontier += p * sub_front
            hit = hit or sub_hit
            if sub_best is not None:
                cand = (p * sub_best[0], (edge.id,) + sub_best[1])
                if better_best(cand, best):
                    best = cand

        memo[key] = (likelihood, visits, best, frontier, hit)
        return memo[key]

    return solve(model.initial, 0, MAX_STEPS)


def silent_graph_is_dag(model):
    adj = {s: [] for s in model.states}
    for e in model.edges:
        if e.label is None and e.prob > 0:
            adj[e.source].append(e.target)
    color = {s: 0 for s in model.states}  # 0 white, 1 gray, 2 black

    def visit(u):
        color[u] = 1
        for v in adj[u]:
            if color[v] == 1:
                return False
            if color[v] == 0 and not visit(v):
                return False
        color[u] = 2
        return True

    return all(visit(s) for s in model.states if color[s] == 0)


def random_model(rng):
    state_count = rng.randint(2, 5)
    states = [f"s{i}" for i in range(state_count)]
    final_count = rng.randint(1, state_count - 1)
    final = set(rng.sample(states, final_count))
    non_final = [s for s in states if s not in final]
    initial = rng.choice(states)
    edges = []
    eid_counter = 0
    allow_cycles = rng.random() < 0.5
    for u in non_final:
        k = rng.randint(0, 3)
        weights = [rng.randint(0, 4) for _ in range(k)]
        denom = rng.choice([1, 2, 3, 4, 6])
        for w in weights:
            v = rng.choice(states)
            label = rng.choice([None, None, "A", "B"])
            if label is None and not allow_cycles and states.index(v) <= states.index(u):
                # keep the silent subgraph acyclic: silent edges go upward
                v = states[min(state_count - 1, states.index(u) + 1)]
            eid = f"e{eid_counter}"
            eid_counter += 1
            edges.append({
                "id": eid, "source": u, "target": v,
                "label": label, "prob": f"{w}/{denom}",
            })
    obs_len = rng.randint(0, 3)
    obs = [rng.choice(["A", "B"]) for _ in range(obs_len)]
    return {
        "states": states,
        "initial": initial,
        "final": sorted(final),
        "transitions": edges,
        "observations": obs,
    }


def hand_crafted_cyclic_cases():
    """Deterministic models containing silent cycles with known values."""
    # 1. Canonical loop, plus an extra labeled-only edge parallel to the exit.
    yield {
        "states": ["s", "f", "g"],
        "initial": "s",
        "final": ["f", "g"],
        "transitions": [
            {"id": "loop", "source": "s", "target": "s", "label": None, "prob": "1/2"},
            {"id": "exitA", "source": "s", "target": "f", "label": "A", "prob": "1/4"},
            {"id": "exitB", "source": "s", "target": "g", "label": "A", "prob": "1/4"},
        ],
        "observations": ["A"],
    }
    # 2. Two-stage silent loops around two observations.
    yield {
        "states": ["s", "u", "f"],
        "initial": "s",
        "final": ["f"],
        "transitions": [
            {"id": "l1", "source": "s", "target": "s", "label": None, "prob": "1/3"},
            {"id": "eA", "source": "s", "target": "u", "label": "A", "prob": "2/3"},
            {"id": "l2", "source": "u", "target": "u", "label": None, "prob": "3/5"},
            {"id": "eB", "source": "u", "target": "f", "label": "B", "prob": "2/5"},
        ],
        "observations": ["A", "B"],
    }
    # 3. Probability-1 closed silent loop, unreachable exit state.
    yield {
        "states": ["s", "f"],
        "initial": "s",
        "final": ["f"],
        "transitions": [
            {"id": "trap", "source": "s", "target": "s", "label": None, "prob": "1/1"},
        ],
        "observations": ["A"],
    }
    # 4. Silent loop whose exit is itself silent, empty observations.
    yield {
        "states": ["s", "t", "f"],
        "initial": "s",
        "final": ["f"],
        "transitions": [
            {"id": "loop", "source": "s", "target": "s", "label": None, "prob": "1/4"},
            {"id": "st", "source": "s", "target": "t", "label": None, "prob": "1/2"},
            {"id": "tf", "source": "t", "target": "f", "label": None, "prob": "1/1"},
        ],
        "observations": [],
    }
    # 5. Silent cycle across two states (not a self-loop).
    yield {
        "states": ["s", "t", "f"],
        "initial": "s",
        "final": ["f"],
        "transitions": [
            {"id": "s2t", "source": "s", "target": "t", "label": None, "prob": "1/2"},
            {"id": "t2s", "source": "t", "target": "s", "label": None, "prob": "1/2"},
            {"id": "eA", "source": "s", "target": "f", "label": "A", "prob": "1/2"},
        ],
        "observations": ["A"],
    }
    # 6. Two-state silent cycle with failure mass, empty observations.
    yield {
        "states": ["s", "t", "f"],
        "initial": "s",
        "final": ["f"],
        "transitions": [
            {"id": "a", "source": "s", "target": "t", "label": None, "prob": "1/2"},
            {"id": "b", "source": "s", "target": "f", "label": None, "prob": "1/4"},
            {"id": "c", "source": "t", "target": "s", "label": None, "prob": "1/2"},
            {"id": "d", "source": "t", "target": "f", "label": None, "prob": "1/2"},
        ],
        "observations": [],
    }
    # 7. The same cycle behind a deterministic labeled edge.
    yield {
        "states": ["s0", "s", "t", "f"],
        "initial": "s0",
        "final": ["f"],
        "transitions": [
            {"id": "pre", "source": "s0", "target": "s", "label": "A", "prob": "1/1"},
            {"id": "a", "source": "s", "target": "t", "label": None, "prob": "1/2"},
            {"id": "b", "source": "s", "target": "f", "label": None, "prob": "1/4"},
            {"id": "c", "source": "t", "target": "s", "label": None, "prob": "1/2"},
            {"id": "d", "source": "t", "target": "f", "label": None, "prob": "1/2"},
        ],
        "observations": ["A"],
    }


def main():
    rng = random.Random(20261112)
    exact_compared = 0
    interval_compared = 0
    cases = [random_model(rng) for _ in range(TRIALS)]
    cases.extend(hand_crafted_cyclic_cases())
    for data in cases:
        try:
            model = parse_model(data)
        except Exception:
            continue  # skip accidentally invalid random inputs
        result = infer(model)
        bf_l, bf_visits, bf_best, frontier, hit_bound = brute_force(model)
        cyclic = not silent_graph_is_dag(model)

        if result.likelihood == 0:
            assert bf_l == 0, (data, bf_l)
            exact_compared += 1
            continue

        if not cyclic and not hit_bound:
            # Fully finite enumeration: exact equality on every quantity.
            assert result.likelihood == bf_l, (
                "likelihood mismatch", data, result.likelihood, bf_l)
            for eid, frac in result.expected_visits.items():
                assert frac == bf_visits[eid] / bf_l, (
                    "expectation mismatch", data, eid,
                    frac, bf_visits[eid] / bf_l)
            ids = tuple(result.best_path["transition_ids"])
            p, q = result.best_path["weight"].split("/")
            assert bf_best == (Fraction(int(p), int(q)), ids), (
                "best path mismatch", data, result.best_path, bf_best)
            exact_compared += 1
        else:
            # Cyclic model: brute force is a truncated lower bound; the
            # frontier bounds every omitted continuation.
            assert bf_l <= result.likelihood <= bf_l + frontier, (
                "likelihood outside truncation interval", data,
                result.likelihood, bf_l, frontier)
            # The best path is cycle-free and therefore fully enumerated.
            ids = tuple(result.best_path["transition_ids"])
            assert len(ids) <= (len(model.observations) + 1) * len(model.states)
            p, q = result.best_path["weight"].split("/")
            assert bf_best == (Fraction(int(p), int(q)), ids), (
                "best path mismatch under cycles", data,
                result.best_path, bf_best)
            # Truncated enumeration can only undercount edge traversals:
            # every enumerated edge count is a lower bound of the exact
            # success-weighted (unconditional) traversal count.
            for eid, frac in result.expected_visits.items():
                assert bf_visits[eid] <= frac * result.likelihood, (
                    "expectation lower-bound violation", data, eid,
                    bf_visits[eid], frac * result.likelihood)
            interval_compared += 1

    print(f"cross-check passed: {exact_compared} exact (finite) models, "
          f"{interval_compared} cyclic models with truncation intervals "
          f"(depth bound {MAX_STEPS})")


if __name__ == "__main__":
    main()
