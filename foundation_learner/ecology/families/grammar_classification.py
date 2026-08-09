"""``grammar_classification`` — membership in a hidden regular language.

Mechanism: the hidden language is a Boolean combination of two atomic
predicates over the alphabet {a, b, c}.  There are eight predicate SCHEMAS,
each parameterised (so the atomic-predicate pool has roughly sixty members):

``count_mod``, ``ends_with``, ``starts_with``, ``contains_bigram``,
``length_mod``, ``no_occurrence``, ``first_before`` and ``count_at_least``.

Each atom may be negated and the two atoms are joined by a hidden ``AND`` or
``OR``.  Every predicate is regular, so the hidden language is regular.  The
task is to classify a string.  Answer: ``IN``/``OUT``.

Difficulty 0/1/2 -> string length 5/8/11.

TRANSFER SHIFT: much longer strings (length 18) where the modular-count and
bigram predicates behave very differently from the short training strings.  The
latent language is unchanged.

Structured feedback: WHICH of the two hidden atomic predicates the string fails,
by opaque predicate index (``A``, ``B``, ``BOTH`` or ``ABSENT``).  The
membership bit itself is never stated.
"""
from __future__ import annotations

import numpy as np

from ..base import (KIND_TRANSFER, Feedback, HintField, Item, Rule, TaskFamily,
                    TaskInstance, derive_seed, make_rng)
from ..surface_remap import PromptSpec, lab, syms

SYMBOL_POOL = ("a", "b", "c", "d", "e", "f")
N_SYMBOLS = 3
SCHEMAS = ("count_mod", "ends_with", "starts_with", "contains_bigram",
           "length_mod", "no_occurrence", "first_before", "count_at_least")
_LENGTHS = {0: 5, 1: 8, 2: 11}
_TRANSFER_LENGTH = 18


def _sample_atom(rng: np.random.Generator) -> dict:
    schema = SCHEMAS[int(rng.integers(0, len(SCHEMAS)))]
    alphabet = SYMBOL_POOL[:N_SYMBOLS]
    if schema == "count_mod":
        m = int(rng.integers(2, 4))
        return {"schema": schema, "sym": alphabet[int(rng.integers(0, N_SYMBOLS))],
                "m": m, "r": int(rng.integers(0, m))}
    if schema in ("ends_with", "starts_with", "no_occurrence"):
        return {"schema": schema,
                "sym": alphabet[int(rng.integers(0, N_SYMBOLS))]}
    if schema == "contains_bigram":
        return {"schema": schema,
                "x": alphabet[int(rng.integers(0, N_SYMBOLS))],
                "y": alphabet[int(rng.integers(0, N_SYMBOLS))]}
    if schema == "length_mod":
        m = int(rng.integers(2, 4))
        return {"schema": schema, "m": m, "r": int(rng.integers(0, m))}
    if schema == "first_before":
        x, y = (int(i) for i in rng.choice(N_SYMBOLS, size=2, replace=False))
        return {"schema": schema, "x": alphabet[x], "y": alphabet[y]}
    return {"schema": schema,
            "sym": alphabet[int(rng.integers(0, N_SYMBOLS))],
            "t": int(rng.integers(1, 3))}


def _holds(atom: dict, text: str) -> bool:
    schema = atom["schema"]
    if schema == "count_mod":
        return text.count(atom["sym"]) % atom["m"] == atom["r"]
    if schema == "ends_with":
        return text.endswith(atom["sym"])
    if schema == "starts_with":
        return text.startswith(atom["sym"])
    if schema == "contains_bigram":
        return (atom["x"] + atom["y"]) in text
    if schema == "length_mod":
        return len(text) % atom["m"] == atom["r"]
    if schema == "no_occurrence":
        return atom["sym"] not in text
    if schema == "first_before":
        xi = text.find(atom["x"])
        yi = text.find(atom["y"])
        return xi != -1 and yi != -1 and xi < yi
    return text.count(atom["sym"]) >= atom["t"]


class GrammarClassificationFamily(TaskFamily):
    family_id = "grammar_classification"
    canon_mode = "label"
    symbol_pool = SYMBOL_POOL
    label_variants = (("ACCEPT", "REJECT"), ("MEMBER", "NONMEMBER"),
                      ("GREEN", "AMBER"))

    def sample_rule(self, rng: np.random.Generator) -> Rule:
        atoms = [_sample_atom(rng), _sample_atom(rng)]
        negate = [int(rng.integers(0, 2)), int(rng.integers(0, 2))]
        connective = "AND" if int(rng.integers(0, 2)) == 0 else "OR"
        return Rule(self.family_id, {"atoms": atoms, "negate": negate,
                                     "connective": connective})

    def _atom_values(self, rule: Rule, text: str) -> list[bool]:
        return [bool(_holds(atom, text)) != bool(neg)
                for atom, neg in zip(rule.params["atoms"], rule.params["negate"])]

    def _member(self, rule: Rule, text: str) -> bool:
        left, right = self._atom_values(rule, text)
        return (left and right) if rule.params["connective"] == "AND" \
            else (left or right)

    # -- items --------------------------------------------------------------

    def _item(self, rule: Rule, raw_seed: int, kind: int, difficulty: int) -> Item:
        rng = make_rng(derive_seed(raw_seed, self.family_id, kind, difficulty))
        length = _TRANSFER_LENGTH if kind == KIND_TRANSFER else _LENGTHS[difficulty]
        alphabet = SYMBOL_POOL[:N_SYMBOLS]
        text = "".join(alphabet[int(i)] for i in
                       rng.integers(0, N_SYMBOLS, size=length))
        member = self._member(rule, text)
        clauses = [
            "A hidden language accepts some strings over the alphabet below.",
            f"Alphabet: {syms(alphabet)}",
            f"String: {syms(text, sep='')}",
        ]
        spec = PromptSpec(
            clauses=tuple(clauses),
            question="Is this string in the hidden language?",
            answer_format=(f"Reply with a final line: ANSWER: {lab('IN')} "
                           f"or {lab('OUT')}"),
            alphabet=tuple(alphabet),
            labels=("IN", "OUT"),
            answer=lab("IN" if member else "OUT"),
        )
        return Item(spec=spec, data={"text": text, "member": member})

    # -- feedback -----------------------------------------------------------

    def _feedback(self, rule: Rule, item: Item, plan, attempt, truth,
                  rng: np.random.Generator) -> Feedback:
        left, right = self._atom_values(rule, item.data["text"])
        if not left and not right:
            value = 2
        elif not left:
            value = 0
        elif not right:
            value = 1
        else:
            value = 3
        return Feedback("HINT-PRED", (
            HintField("fails", "PRD", value, 4, ("A", "B", "BOTH", "ABSENT")),))

    def plausible_error(self, rule: Rule, instance: TaskInstance,
                        rng: np.random.Generator) -> str:
        return self._flip_label(rule, instance)


FAMILY = GrammarClassificationFamily()
