"""``constraint_rules`` — satisfaction checking against a hidden constraint set.

Mechanism: a hidden constraint template over 3-5 named slots.  Every ordered
slot pair (i, j) carries one of ``LT`` (slot_i < slot_j), ``GT``, ``NEQ`` or no
constraint, and each slot may additionally carry a hidden forbidden value.  The
task is to decide whether a displayed full assignment satisfies ALL hidden
constraints.  Answer: ``SAT``/``UNSAT``.

Difficulty 0/1/2 -> assignment values drawn from 1..4 / 1..6 / 1..8.

TRANSFER SHIFT: the assignment is presented over EXTRA slots that the hidden
constraint set never mentions, drawn from a wider value range, so the surface
assignment is longer and noisier than any training instance.  The latent
constraint set is unchanged.

Structured feedback: the number of violated constraints, a count only.  The
satisfaction bit itself is never stated (a count of zero is of course
informative — that is the point of informative feedback — but it is never the
answer token).
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, make_rng)
from ..surface_remap import PromptSpec, lab, sym

SLOT_POOL = ("sl", "sm", "sn", "so", "sp", "sq", "sr", "ss", "st")
COMPARATORS = ("none", "LT", "GT", "NEQ")
_VALUE_MAX = {0: 5, 1: 7, 2: 9}
_TRANSFER_VALUE_MAX = 12
_TRANSFER_EXTRA_SLOTS = 3


class ConstraintRulesFamily(TaskFamily):
    family_id = "constraint_rules"
    canon_mode = "label"
    symbol_pool = SLOT_POOL
    label_variants = (("VALID", "INVALID"), ("HOLDS", "FAILS"),
                      ("PASS", "REJECT"))

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        n_slots = int(rng.integers(3, 6))
        pairs = []
        for i in range(n_slots):
            for j in range(i + 1, n_slots):
                comparator = COMPARATORS[int(rng.integers(0, len(COMPARATORS)))]
                if comparator != "none":
                    pairs.append([i, j, comparator])
        forbidden = []
        for i in range(n_slots):
            if int(rng.integers(0, 3)) == 0:
                forbidden.append([i, int(rng.integers(1, 6))])
        if not pairs and not forbidden:
            pairs.append([0, 1, "LT"])
        return Rule(self.family_id, {"n_slots": n_slots, "pairs": pairs,
                                     "forbidden": forbidden})

    @staticmethod
    def _violations(rule: Rule, values: list[int]) -> int:
        count = 0
        for i, j, comparator in rule.params["pairs"]:
            a, b = values[i], values[j]
            if comparator == "LT" and not a < b:
                count += 1
            elif comparator == "GT" and not a > b:
                count += 1
            elif comparator == "NEQ" and not a != b:
                count += 1
        for i, bad in rule.params["forbidden"]:
            if values[i] == bad:
                count += 1
        return count

    def _n_constraints(self, rule: Rule) -> int:
        return len(rule.params["pairs"]) + len(rule.params["forbidden"])

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        n_slots = rule.params["n_slots"]
        transfer = kind == KIND_TRANSFER
        hi = _TRANSFER_VALUE_MAX if transfer else _VALUE_MAX[difficulty]
        shown = min(n_slots + (_TRANSFER_EXTRA_SLOTS if transfer else 0),
                    len(SLOT_POOL))
        values = [int(v) for v in rng.integers(1, hi, size=shown)]
        violations = self._violations(rule, values[:n_slots])
        answer = "SAT" if violations == 0 else "UNSAT"

        assignment = ", ".join(f"{sym(SLOT_POOL[i])}={values[i]}"
                               for i in range(shown))
        clauses = [
            "A hidden set of constraints restricts how the slots may be filled.",
            f"Assignment: {assignment}",
        ]
        if transfer:
            clauses.append("Some slots shown here may not be constrained at all.")
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="Does this assignment satisfy every hidden constraint?",
            answer_format=(f"Reply with a final line: ANSWER: {lab('SAT')} "
                           f"or {lab('UNSAT')}"),
            alphabet=tuple(SLOT_POOL[:shown]),
            labels=("SAT", "UNSAT"),
            answer=lab(answer),
        )
        return Item(spec=spec, data={"values": values, "shown": shown,
                                     "violations": violations})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        total = self._n_constraints(rule)
        return Feedback("HINT-VIOL", (
            HintField("violated", "CNT", item.data["violations"], total + 1),))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        return self._flip_label(rule, instance)


FAMILY = ConstraintRulesFamily()
