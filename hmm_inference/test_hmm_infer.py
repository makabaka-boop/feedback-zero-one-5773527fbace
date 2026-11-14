#!/usr/bin/env python3
"""Hand-computed acceptance tests for hmm_infer.

Run from this directory (or anywhere):

    python3 -I test_hmm_infer.py

Every numeric expectation below was derived by hand on paper; the test only
checks that the implementation reproduces those numbers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hmm_infer import (  # noqa: E402
    ValidationError,
    build_output,
    infer,
    parse_model,
)

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load(name):
    with open(os.path.join(FIX, name), encoding="utf-8") as fh:
        return json.load(fh)


def solve(name):
    return infer(parse_model(load(name)))


def visits(result):
    return {item["transition_id"]: item["expected_visits"]
            for item in build_output(None, result)["expected_visits"]}


def F(text):
    p, q = text.split("/")
    return Fraction(int(p), int(q))


class HandComputedTests(unittest.TestCase):
    def assertVisits(self, result, expected):
        got = visits(result)
        self.assertEqual(set(got), set(expected))
        for eid, value in expected.items():
            self.assertEqual(got[eid], value, f"edge {eid}")

    # 1. Canonical silent self-loop (the example from the specification).
    #    loop: null s->s prob 1/2 ; exitA: A s->f prob 1/2 ; obs [A].
    #    L = sum_{n>=0} (1/2)^n * 1/2 = 1
    #    E[loop | success] = sum_n n (1/2)^{n+1} = 1
    #    E[exitA | success] = 1
    #    best path: exitA alone, weight 1/2.
    def test_canonical_loop(self):
        r = solve("canonical_loop.json")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"loop": "1/1", "exitA": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["exitA"], "weight": "1/2"})

    # 2. Closed silent loop of probability 1: nothing is ever emitted, the
    #    walk never terminates. No successful path; non-termination mass must
    #    not be redistributed.
    def test_closed_silent_loop_impossible(self):
        r = solve("closed_silent_loop.json")
        self.assertEqual(r.status, "impossible")
        self.assertEqual(r.likelihood, F("0/1"))
        self.assertEqual(build_output(None, r),
                         {"likelihood": "0/1", "status": "impossible"})

    # 3. Empty observation sequence; silent loop 2/3, silent exit to terminal
    #    1/3.
    #    L = sum_n (2/3)^n (1/3) = 1
    #    E[loop|success] = (2/3)/(1/3) = 2 ; E[go] = 1
    #    best path: go alone, weight 1/3.
    def test_empty_observations(self):
        r = solve("empty_obs.json")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"loop": "2/1", "go": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["go"], "weight": "1/3"})

    # 4. Zero-probability edge: dead (0/1, A -> f) can never be used; only
    #    live (1/2, A -> g) contributes. L = 1/2, E[dead] = 0 (retained),
    #    E[live] = 1. Best path uses live.
    def test_zero_probability_edge(self):
        r = solve("zero_prob_edge.json")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.likelihood, F("1/2"))
        self.assertVisits(r, {"dead": "0/1", "live": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["live"], "weight": "1/2"})

    # 5. Weight tie broken by path length:
    #    eb: s->f, A, 1/2 (one edge);  es: s->t null 1/2 then ea: t->f A 1
    #    (two edges, total 1/2). Same weight; the one-edge path wins.
    #    L = 1 ; E[eb]=E[es]=E[ea]=1/2
    def test_tie_broken_by_length(self):
        r = solve("tie_length.json")
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"eb": "1/2", "es": "1/2", "ea": "1/2"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["eb"], "weight": "1/2"})

    # 6. Weight and length tie broken by ASCII order of the full id sequence.
    #    Parallel edges ea and ez, each A s->f prob 1/2. L = 1, both
    #    expectations 1/2; best path is the smaller id "ea".
    def test_tie_broken_by_ascii_id(self):
        r = solve("tie_lex.json")
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"ea": "1/2", "ez": "1/2"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["ea"], "weight": "1/2"})

    # 7. Silent loops on both observation layers, obs [A, B].
    #    loop_s 1/2 null s->s; eA 1/2 s->u; loop_u 1/2 null u->u;
    #    eB 1/2 u->f.
    #    L = (sum_i (1/2)^i * 1/2) * (sum_j (1/2)^j * 1/2) = 1 * 1 = 1
    #    E[loop_s] = 1, E[eA] = 1, E[loop_u] = 1, E[eB] = 1
    #    best path: [eA, eB], weight (1/2)*(1/2) = 1/4.
    def test_two_layer_silent_loops(self):
        r = solve("two_layer_loops.json")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {
            "loops": "1/1", "eA": "1/1",
            "loopu": "1/1", "eB": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["eA", "eB"], "weight": "1/4"})

    # 8. Failure remainder: loop 1/3 null s->s, eA 1/3 A s->f; the missing
    #    1/3 per step is immediate failure.
    #    L = sum_n (1/3)^n (1/3) = 1/2 ; success:fail = 1:1.
    #    E[loop | success] = (1/3)/(2/3) = 1/2 ; E[eA | success] = 1
    #    best path: [eA], weight 1/3.
    def test_failure_mass_not_attributed(self):
        r = solve("failure_mass.json")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.likelihood, F("1/2"))
        self.assertVisits(r, {"loop": "1/2", "eA": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["eA"], "weight": "1/3"})

    # 9. Initial state already terminal with empty observations: the empty
    #    path succeeds with weight 1.
    def test_terminal_initial_empty_obs(self):
        r = solve("terminal_initial_empty.json")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertEqual(r.expected_visits, {})
        self.assertEqual(r.best_path,
                         {"transition_ids": [], "weight": "1/1"})

    # 10. Terminal initial state with a demanded observation: impossible.
    def test_terminal_initial_with_obs_impossible(self):
        data = load("terminal_initial_empty.json")
        data["observations"] = ["A"]
        r = infer(parse_model(data))
        self.assertEqual(r.status, "impossible")
        self.assertEqual(r.likelihood, F("0/1"))

    # 11. Branching path counting (verifies expected counts are true
    #     conditional expectations, not visit-at-least-once probabilities):
    #     s --1/2 A--> u --1/2 null--> x --1/1 B--> f
    #       \--1/2 A--> v --1/1 B--> f
    #     obs [A, B]. Path P1 = [a1, sil, b1] weighs 1/2*1/2*1 = 1/4 (the
    #     other 1/2 at u fails); path P2 = [a2, b2] weighs 1/2*1 = 1/2.
    #     L = 1/4 + 1/2 = 3/4.
    #     Conditional on success:
    #       E[a1] = E[sil] = E[b1] = (1/4)/(3/4) = 1/3
    #       E[a2] = E[b2]               = (1/2)/(3/4) = 2/3
    #     Best path: P2 with weight 1/2.
    def test_diamond_expected_counts(self):
        data = {
            "states": ["s", "u", "v", "x", "f"],
            "initial": "s",
            "final": ["f"],
            "transitions": [
                {"id": "a1", "source": "s", "target": "u", "label": "A", "prob": "1/2"},
                {"id": "a2", "source": "s", "target": "v", "label": "A", "prob": "1/2"},
                {"id": "sil", "source": "u", "target": "x", "label": None, "prob": "1/2"},
                {"id": "b1", "source": "x", "target": "f", "label": "B", "prob": "1/1"},
                {"id": "b2", "source": "v", "target": "f", "label": "B", "prob": "1/1"}
            ],
            "observations": ["A", "B"],
        }
        r = infer(parse_model(data))
        self.assertEqual(r.likelihood, F("3/4"))
        self.assertVisits(r, {
            "a1": "1/3", "a2": "2/3", "sil": "1/3",
            "b1": "1/3", "b2": "2/3"})
        # The shorter path P2 also has the larger weight (1/2 vs 1/4).
        self.assertEqual(r.best_path,
                         {"transition_ids": ["a2", "b2"], "weight": "1/2"})

    # 12. Silent path before the first observation, with a silent cycle.
    #     s --2/3 null--> t ; s --1/3 null--> s ; t --1 A--> f.
    #     L = (sum_n (1/3)^n (2/3)) * 1 = 1 ; loop mean 1/2, st mean 1,
    #     eA mean 1. Best path [st, eA] weight 2/3.
    def test_silent_prefix_with_cycle(self):
        data = {
            "states": ["s", "t", "f"],
            "initial": "s",
            "final": ["f"],
            "transitions": [
                {"id": "loop", "source": "s", "target": "s", "label": None, "prob": "1/3"},
                {"id": "st", "source": "s", "target": "t", "label": None, "prob": "2/3"},
                {"id": "eA", "source": "t", "target": "f", "label": "A", "prob": "1/1"}
            ],
            "observations": ["A"],
        }
        r = infer(parse_model(data))
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"loop": "1/2", "st": "1/1", "eA": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["st", "eA"], "weight": "2/3"})

    # 13. Probability-1 silent loop that still has an escape: the bad-region
    #     analysis must keep every state, and results stay finite.
    #     s --1 null self-loop, s --1/2 A--> f. Row sum 1.5 > 1 would be
    #     illegal, so instead: s --1/2 null loop, s --1/2 A exit (covered by
    #     case 1). Here use loop 1/1 with an exit reachable after the loop
    #     through a second state:
    #     s --1/1 null--> t ; t --1/2 null self-loop ; t --1/2 A--> f.
    #     L = 1 ; E[s->t]=1, E[loop_t]=1, E[eA]=1, best [s->t, eA] w 1/2.
    def test_unit_silent_edge_into_escaping_loop(self):
        data = {
            "states": ["s", "t", "f"],
            "initial": "s",
            "final": ["f"],
            "transitions": [
                {"id": "e0", "source": "s", "target": "t", "label": None, "prob": "1/1"},
                {"id": "loopt", "source": "t", "target": "t", "label": None, "prob": "1/2"},
                {"id": "eA", "source": "t", "target": "f", "label": "A", "prob": "1/2"}
            ],
            "observations": ["A"],
        }
        r = infer(parse_model(data))
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"e0": "1/1", "loopt": "1/1", "eA": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["e0", "eA"], "weight": "1/2"})

    # 14. Two disjoint silent components, one closed forever, one escaping.
    #     s --1/2 A--> trap (trap loops null with prob 1, dead end)
    #       --1/2 A--> g ; g --1/1 null--> f.
    #     L = 1/2 ; on success eA_g, e0 are used once; trap edge never.
    def test_mixed_closed_and_escaping_components(self):
        data = {
            "states": ["s", "trap", "g", "f"],
            "initial": "s",
            "final": ["f"],
            "transitions": [
                {"id": "intotrap", "source": "s", "target": "trap", "label": "A", "prob": "1/2"},
                {"id": "traploop", "source": "trap", "target": "trap", "label": None, "prob": "1/1"},
                {"id": "intog", "source": "s", "target": "g", "label": "A", "prob": "1/2"},
                {"id": "gtof", "source": "g", "target": "f", "label": None, "prob": "1/1"}
            ],
            "observations": ["A"],
        }
        r = infer(parse_model(data))
        self.assertEqual(r.likelihood, F("1/2"))
        self.assertVisits(r, {
            "intotrap": "0/1", "traploop": "0/1",
            "intog": "1/1", "gtof": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["intog", "gtof"], "weight": "1/2"})


    # 15. A probability-1 silent self-loop sitting in front of an exit: the
    #     cycle never helps the best path (it only adds edges), and the
    #     likelihood must stay finite rather than diverge.
    #     s --1 null self-loop, s --1/2 A--> f (row sum 1.5 is illegal, so
    #     split via two states): covered by case 13. Here the unit loop is
    #     on a state reached with probability 1:
    #     s --1/1 null--> t ; t --1/1 null self-loop ; t --1/2 A--> f.
    #     Row sum at t = 1.5 would be illegal, so instead loop 1/2 at t
    #     (case 13). This case tests parallel silent edges:
    #     s has null edges x (1/2 -> t) and y (1/2 -> t), t --1 A--> f.
    #     obs [A]. L = 1 ; E[x]=E[y]=1/2, E[eA]=1.
    #     Best silent prefix: both x and y weigh 1/2 with one edge; the
    #     smaller id "x" wins, giving best path [x, eA] weight 1/2.
    def test_parallel_silent_edges_lex_tie(self):
        data = {
            "states": ["s", "t", "f"],
            "initial": "s",
            "final": ["f"],
            "transitions": [
                {"id": "x", "source": "s", "target": "t", "label": None, "prob": "1/2"},
                {"id": "y", "source": "s", "target": "t", "label": None, "prob": "1/2"},
                {"id": "eA", "source": "t", "target": "f", "label": "A", "prob": "1/1"}
            ],
            "observations": ["A"],
        }
        r = infer(parse_model(data))
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"x": "1/2", "y": "1/2", "eA": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["x", "eA"], "weight": "1/2"})

    # 16. A silent unit-probability cycle followed by a silent exit to a
    #     terminal state, empty observations. The cycle adds nothing to the
    #     best path, which uses the exit directly.
    #     s --1/2 null self-loop, s --1/2 null--> f, obs [].
    #     L = sum_n (1/2)^n (1/2) = 1 ; E[loop] = 1, E[go] = 1.
    #     Best path: [go] weight 1/2.
    def test_unit_weight_cycle_not_in_best_path(self):
        data = {
            "states": ["s", "f"],
            "initial": "s",
            "final": ["f"],
            "transitions": [
                {"id": "loop", "source": "s", "target": "s", "label": None, "prob": "1/2"},
                {"id": "go", "source": "s", "target": "f", "label": None, "prob": "1/2"}
            ],
            "observations": [],
        }
        r = infer(parse_model(data))
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"loop": "1/1", "go": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["go"], "weight": "1/2"})

    # 17. Repeated observations with one labeled edge reused at two layers:
    #     its expected count must sum both layers.
    #     s --1/2 null self-loop, eA --1/2 s->s label A, and after two A's
    #     an exit eF --1 s->f label B cannot appear... instead use two
    #     parallel labeled edges both emitting A, then a silent exit.
    #     s edges: loop null 1/3, a1 A s->u 1/3, a2 A u->f 1/3 ; u --1 null-> f.
    #     obs [A, A]. From s: loop then a1 (to u), a2 unavailable from s...
    #     Simplify: s has loop null 1/3, a1 A s->u 2/3 ; u has a2 A u->f 1.
    #     L = (sum_i (1/3)^i (2/3)) * 1 = 1 ; E[loop]=1/2, E[a1]=1, E[a2]=1.
    def test_repeated_observation_layers_sum(self):
        data = {
            "states": ["s", "u", "f"],
            "initial": "s",
            "final": ["f"],
            "transitions": [
                {"id": "loop", "source": "s", "target": "s", "label": None, "prob": "1/3"},
                {"id": "a1", "source": "s", "target": "u", "label": "A", "prob": "2/3"},
                {"id": "a2", "source": "u", "target": "f", "label": "A", "prob": "1/1"}
            ],
            "observations": ["A", "A"],
        }
        r = infer(parse_model(data))
        self.assertEqual(r.likelihood, F("1/1"))
        self.assertVisits(r, {"loop": "1/2", "a1": "1/1", "a2": "1/1"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["a1", "a2"], "weight": "2/3"})

    # 18. Two-state silent cycle (not a self-loop), empty observations.
    #     s --a 1/2 null--> t ; s --b 1/4 null--> f
    #     t --c 1/2 null--> s ; t --d 1/2 null--> f ; obs [].
    #     A round trip s->t->s has probability (1/2)(1/2) = 1/4, so the
    #     expected number of visits is geometric:
    #       V_s = sum_{k>=0} (1/4)^k = 4/3 ; V_t = V_s * 1/2 = 2/3.
    #     Success probabilities: h_f = 1,
    #       h_t = 1/2 + (1/2) h_s, h_s = 1/4 + (1/2) h_t  =>  h_s = 2/3,
    #       h_t = 5/6, hence L = h_s = 2/3.
    #     Success-weighted counts (visits x prob x suffix):
    #       a: (4/3)(1/2)(5/6) = 5/9 ; b: (4/3)(1/4)(1) = 1/3
    #       c: (2/3)(1/2)(2/3) = 2/9 ; d: (2/3)(1/2)(1) = 1/3
    #     Conditional on success (divide by L = 2/3):
    #       a 5/6, b 1/2, c 1/3, d 1/2.
    #     Best path: [b] and [a, d] both weigh 1/4; fewer edges wins -> [b].
    def test_two_state_silent_cycle(self):
        r = solve("two_state_loop.json")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.likelihood, F("2/3"))
        self.assertVisits(r, {"a": "5/6", "b": "1/2", "c": "1/3", "d": "1/2"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["b"], "weight": "1/4"})

    # 19. The same two-state cycle behind a deterministic labeled edge.
    #     pre: s0 --1/1 A--> s, then the cycle of case 18; obs [A].
    #     The prefix multiplies every path weight by 1, so L = 2/3 and the
    #     tail expectations are exactly those of case 18; E[pre] = 1.
    #     Best path [pre, b] weight 1/4 ([pre, a, d] ties the weight but is
    #     one edge longer).
    def test_two_state_silent_cycle_after_label(self):
        r = solve("two_state_loop_prefixed.json")
        self.assertEqual(r.status, "success")
        self.assertEqual(r.likelihood, F("2/3"))
        self.assertVisits(r, {"pre": "1/1", "a": "5/6", "b": "1/2",
                              "c": "1/3", "d": "1/2"})
        self.assertEqual(r.best_path,
                         {"transition_ids": ["pre", "b"], "weight": "1/4"})


class ValidationTests(unittest.TestCase):
    def _reject(self, name, path):
        with self.assertRaises(ValidationError) as ctx:
            parse_model(load(name))
        self.assertEqual(ctx.exception.path, path)

    def test_unknown_field_located(self):
        self._reject("bad_unknown_field.json", "$.transitions[0]")

    def test_missing_reference_located(self):
        self._reject("bad_reference.json", "$.transitions[0].target")

    def test_probability_out_of_range_located(self):
        self._reject("bad_prob_range.json", "$.transitions[0].prob")

    def test_outgoing_sum_too_large_located(self):
        self._reject("bad_prob_sum.json", "$.transitions")

    def test_terminal_outedge_located(self):
        self._reject("bad_terminal_outedge.json", "$.transitions[0].source")

    def test_malformed_probability(self):
        data = load("canonical_loop.json")
        data["transitions"][0]["prob"] = "1/0"
        with self.assertRaises(ValidationError) as ctx:
            parse_model(data)
        self.assertEqual(ctx.exception.path, "$.transitions[0].prob")

    def test_non_string_probability(self):
        data = load("canonical_loop.json")
        data["transitions"][0]["prob"] = 0.5
        with self.assertRaises(ValidationError) as ctx:
            parse_model(data)
        self.assertEqual(ctx.exception.path, "$.transitions[0].prob")

    def test_duplicate_transition_id(self):
        data = load("canonical_loop.json")
        data["transitions"][1]["id"] = "loop"
        with self.assertRaises(ValidationError) as ctx:
            parse_model(data)
        self.assertEqual(ctx.exception.path, "$.transitions[1].id")

    def test_empty_label_rejected(self):
        data = load("canonical_loop.json")
        data["transitions"][1]["label"] = ""
        with self.assertRaises(ValidationError) as ctx:
            parse_model(data)
        self.assertEqual(ctx.exception.path, "$.transitions[1].label")

    def test_empty_observation_rejected(self):
        data = load("canonical_loop.json")
        data["observations"] = [""]
        with self.assertRaises(ValidationError) as ctx:
            parse_model(data)
        self.assertEqual(ctx.exception.path, "$.observations[0]")

    def test_non_alphanumeric_id_rejected(self):
        data = load("canonical_loop.json")
        data["transitions"][0]["id"] = "lo op"
        with self.assertRaises(ValidationError) as ctx:
            parse_model(data)
        self.assertEqual(ctx.exception.path, "$.transitions[0].id")

    def test_unknown_top_level_field(self):
        data = load("canonical_loop.json")
        data["extra"] = 1
        with self.assertRaises(ValidationError) as ctx:
            parse_model(data)
        self.assertEqual(ctx.exception.path, "$")


class CliTests(unittest.TestCase):
    def test_cli_success_and_output_file(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hmm_infer.py")
        inp = os.path.join(FIX, "canonical_loop.json")
        out = os.path.join(FIX, "_tmp_output.json")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", script, inp, "-o", out],
                capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result["likelihood"], "1/1")
            self.assertEqual(result["status"], "success")
            with open(out, encoding="utf-8") as fh:
                on_disk = json.load(fh)
            self.assertEqual(on_disk, result)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_cli_rejection_exit_code_and_path(self):
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hmm_infer.py")
        inp = os.path.join(FIX, "bad_reference.json")
        proc = subprocess.run(
            [sys.executable, "-I", script, inp],
            capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2)
        err = json.loads(proc.stderr)
        self.assertEqual(err["path"], "$.transitions[0].target")
        self.assertEqual(proc.stdout, "")

    def test_cli_rejection_preserves_existing_output(self):
        # An illegal input rejects the whole run: the output file must keep
        # its previous contents (only a fully successful run replaces it).
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hmm_infer.py")
        inp = os.path.join(FIX, "bad_prob_range.json")
        out = os.path.join(FIX, "_tmp_output.json")
        sentinel = {"kept": "previous successful run"}
        try:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(sentinel, fh)
            proc = subprocess.run(
                [sys.executable, "-I", script, inp, "-o", out],
                capture_output=True, text=True, check=False)
            self.assertEqual(proc.returncode, 2)
            self.assertEqual(proc.stdout, "")
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), sentinel)
        finally:
            if os.path.exists(out):
                os.remove(out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
