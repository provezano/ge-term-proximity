"""The thesis grammar (Figure 15), derivation trees, genotype mapping and evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Production order matters: GE picks production (codon mod n_choices).
GRAMMAR_BNF = """\
<start> ::= <sc>
<sc>    ::= ((<sc>)<op1>(<sc>))
          | ((<sc>)<op1>(<n><n>.<n><n>))
          | ((<n><n>.<n><n>)<op1><sc>)
          | <op2>(<sc>)
          | (<t2>*<t2>)
          | ((<t2>*<t2>)<op1>(<sc>))
          | <t1>
<op1>   ::= + | - | * | / | ^
<op2>   ::= sin | tanh | sqrt | ln | log2
<t1>    ::= p(ti,tj) | p(ti,~tj) | p(~ti,tj) | p(~ti,~tj)
<t2>    ::= p(ti) | p(~ti) | p(tj) | p(~tj)
<n>     ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
"""

_TOKEN = re.compile(r"<[^<>\s]+>|[^<]+")


def parse_bnf(text: str) -> dict[str, list[list[str]]]:
    rules: dict[str, list[list[str]]] = {}
    current = None
    for line in text.splitlines():
        if "::=" in line:
            lhs, line = line.split("::=")
            current = lhs.strip()
            rules[current] = []
        for alt in line.split("|"):
            alt = alt.replace(" ", "")
            if alt:
                rules[current].append(_TOKEN.findall(alt))
    return rules


GRAMMAR = parse_bnf(GRAMMAR_BNF)
START = "<start>"

TERMINAL_VARS = {
    "p(ti,tj)": "pab", "p(ti,~tj)": "paB", "p(~ti,tj)": "pAb", "p(~ti,~tj)": "pAB",
    "p(ti)": "pa", "p(~ti)": "pA", "p(tj)": "pb", "p(~tj)": "pB",
}


def is_nt(sym: str) -> bool:
    return sym.startswith("<") and sym.endswith(">") and len(sym) > 2


def _heights(rules):
    """Minimum subtree height (in non-terminal levels) of every non-terminal and production."""
    inf = 10**9
    h = {nt: inf for nt in rules}
    changed = True
    while changed:
        changed = False
        for nt, prods in rules.items():
            best = min(1 + max([h[s] for s in p if is_nt(s)], default=0) for p in prods)
            if best < h[nt]:
                h[nt], changed = best, True
    ph = {nt: [1 + max([h[s] for s in p if is_nt(s)], default=0) for p in prods]
          for nt, prods in rules.items()}
    return h, ph


def _recursive(rules):
    """Non-terminals that can derive themselves."""
    reach = {nt: {s for p in prods for s in p if is_nt(s)} for nt, prods in rules.items()}
    changed = True
    while changed:
        changed = False
        for nt in rules:
            new = set().union(*(reach[s] for s in reach[nt])) | reach[nt]
            if new != reach[nt]:
                reach[nt], changed = new, True
    return {nt for nt in rules if nt in reach[nt]}


MIN_HEIGHT, PROD_HEIGHT = _heights(GRAMMAR)
RECURSIVE_NT = _recursive(GRAMMAR)


def production_is_recursive(nt: str, choice: int) -> bool:
    return any(is_nt(s) and s in RECURSIVE_NT for s in GRAMMAR[nt][choice])


@dataclass
class Node:
    symbol: str
    choice: int
    children: list["Node"] = field(default_factory=list)

    def copy(self) -> "Node":
        return Node(self.symbol, self.choice, [c.copy() for c in self.children])

    def nodes(self, depth: int = 1):
        """Pre-order (node, depth) pairs."""
        yield self, depth
        for c in self.children:
            yield from c.nodes(depth + 1)

    def depth(self) -> int:
        return max(d for _, d in self.nodes())

    def phenotype(self) -> str:
        out, kids = [], iter(self.children)
        for s in GRAMMAR[self.symbol][self.choice]:
            out.append(next(kids).phenotype() if is_nt(s) else s)
        return "".join(out)

    def genome(self) -> list[int]:
        """Codons (choices of multi-option rules, pre-order) that map back to this tree."""
        out = [self.choice] if len(GRAMMAR[self.symbol]) > 1 else []
        for c in self.children:
            out.extend(c.genome())
        return out


def map_genome(genome: list[int], max_wraps: int = 0, max_depth: int | None = None):
    """Standard GE genotype-to-phenotype mapping, leftmost non-terminal first.

    Single-option rules consume no codon. Returns None if the genome (with wraps)
    runs out before the derivation is complete, or the tree exceeds ``max_depth``.
    """
    used = 0
    limit = len(genome) * (max_wraps + 1)
    root = Node(START, -1)
    stack = [(root, 1)]
    while stack:
        node, d = stack.pop()
        if max_depth is not None and d > max_depth:
            return None
        n = len(GRAMMAR[node.symbol])
        if n == 1:
            node.choice = 0
        else:
            if used >= limit or not genome:
                return None
            node.choice = genome[used % len(genome)] % n
            used += 1
        node.children = [Node(s, -1) for s in GRAMMAR[node.symbol][node.choice] if is_nt(s)]
        stack.extend((c, d + 1) for c in reversed(node.children))
    return root


_OP1 = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "^": np.power,
}
_OP2 = {"sin": np.sin, "tanh": np.tanh, "sqrt": np.sqrt, "ln": np.log, "log2": np.log2}


def _number(nodes) -> float:
    d = [str(n.choice) for n in nodes]
    return float(f"{d[0]}{d[1]}.{d[2]}{d[3]}")


def _eval(node: Node, s):
    sym = node.symbol
    ch = node.children
    if sym == "<start>":
        return _eval(ch[0], s)
    if sym in ("<t1>", "<t2>"):
        return s[TERMINAL_VARS[GRAMMAR[sym][node.choice][0]]]
    if sym != "<sc>":
        raise ValueError(f"cannot evaluate {sym}")
    c = node.choice
    op = lambda n: _OP1[GRAMMAR["<op1>"][n.choice][0]]  # noqa: E731
    if c == 0:
        return op(ch[1])(_eval(ch[0], s), _eval(ch[2], s))
    if c == 1:
        return op(ch[1])(_eval(ch[0], s), _number(ch[2:6]))
    if c == 2:
        return op(ch[4])(_number(ch[0:4]), _eval(ch[5], s))
    if c == 3:
        return _OP2[GRAMMAR["<op2>"][ch[0].choice][0]](_eval(ch[1], s))
    if c == 4:
        return _eval(ch[0], s) * _eval(ch[1], s)
    if c == 5:
        return op(ch[2])(_eval(ch[0], s) * _eval(ch[1], s), _eval(ch[3], s))
    return _eval(ch[0], s)


def evaluate(tree: Node, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Proximity matrix (n x n) of an evolved function; non-finite values become NaN."""
    n = stats["pab"].shape[0]
    with np.errstate(all="ignore"):
        out = np.broadcast_to(np.asarray(_eval(tree, stats), dtype=np.float64), (n, n))
    return np.where(np.isfinite(out), out, np.nan)
