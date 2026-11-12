"""A small grammatical evolution engine on derivation trees.

Initialisation is position-independent grow (Fagan, Fenton and O'Neill, 2016), variation is
subtree crossover and subtree mutation (as in PonyGE2), selection is tournament, replacement is
generational with elitism. All randomness comes from one ``numpy.random.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grammar import GRAMMAR, MIN_HEIGHT, PROD_HEIGHT, START, Node, is_nt, production_is_recursive


@dataclass
class GEConfig:
    population: int = 100
    generations: int = 100
    crossover_prob: float = 0.75
    mutation_prob: float = 0.10
    tournament: int = 5
    elitism: int = 2
    max_init_depth: int = 10
    max_depth: int = 90


def _choices_that_fit(nt: str, depth: int, max_depth: int) -> list[int]:
    return [i for i, h in enumerate(PROD_HEIGHT[nt]) if depth + h - 1 <= max_depth]


def _expand(node: Node, depth: int, choice: int, queue: list) -> None:
    node.choice = choice
    node.children = [Node(s, -1) for s in GRAMMAR[node.symbol][choice] if is_nt(s)]
    queue.extend((c, depth + 1) for c in node.children)


def pi_grow(rng: np.random.Generator, target_depth: int, symbol: str = START) -> Node:
    """Position-independent grow: expand open non-terminals in random order and keep
    choosing recursive productions until some branch reaches ``target_depth``."""
    root = Node(symbol, -1)
    queue = [(root, 1)]
    reached = False
    while queue:
        node, d = queue.pop(int(rng.integers(len(queue))))
        fits = _choices_that_fit(node.symbol, d, target_depth)
        if not reached:
            rec = [c for c in fits if production_is_recursive(node.symbol, c)]
            fits = rec or fits
        _expand(node, d, int(rng.choice(fits)), queue)
        if d + 1 >= target_depth and node.children:
            reached = True
    return root


def grow(rng: np.random.Generator, symbol: str, depth: int, max_depth: int) -> Node:
    """Random grow of a subtree rooted at ``symbol`` placed at ``depth``."""
    root = Node(symbol, -1)
    queue = [(root, depth)]
    while queue:
        node, d = queue.pop()
        _expand(node, d, int(rng.choice(_choices_that_fit(node.symbol, d, max_depth))), queue)
    return root


def _pick(rng, tree: Node, symbols: set[str] | None = None):
    """Random (parent, index, node, depth) among non-terminal nodes below the root."""
    found = []
    stack = [(tree, 1)]
    while stack:
        node, d = stack.pop()
        for i, c in enumerate(node.children):
            if symbols is None or c.symbol in symbols:
                found.append((node, i, c, d + 1))
            stack.append((c, d + 1))
    return found[int(rng.integers(len(found)))] if found else None


def subtree_mutation(rng, tree: Node, max_depth: int) -> Node:
    tree = tree.copy()
    pick = _pick(rng, tree)
    if pick is not None:
        parent, i, node, d = pick
        parent.children[i] = grow(rng, node.symbol, d, max_depth)
    return tree


def subtree_crossover(rng, a: Node, b: Node, max_depth: int) -> tuple[Node, Node]:
    a, b = a.copy(), b.copy()
    sym_b = {n.symbol for n, _ in b.nodes()}
    pa = _pick(rng, a, sym_b)
    if pa is None:
        return a, b
    pb = _pick(rng, b, {pa[2].symbol})
    if pb is None:
        return a, b
    (par_a, ia, na, _), (par_b, ib, nb, _) = pa, pb
    par_a.children[ia], par_b.children[ib] = nb, na
    if a.depth() > max_depth or b.depth() > max_depth:
        par_a.children[ia], par_b.children[ib] = na, nb
    return a, b


@dataclass
class Individual:
    tree: Node
    fitness: float = float("-inf")

    @property
    def phenotype(self) -> str:
        return self.tree.phenotype()


@dataclass
class RunResult:
    best: Individual
    history: list[float] = field(default_factory=list)
    evaluations: int = 0


def evolve(fitness: Callable[[Node], float], config: GEConfig, seed: int) -> RunResult:
    """Maximise ``fitness`` over derivation trees of the grammar."""
    rng = np.random.default_rng(seed)
    cache: dict[str, float] = {}

    def score(ind: Individual) -> Individual:
        key = ind.phenotype
        if key not in cache:
            f = float(fitness(ind.tree))
            cache[key] = f if np.isfinite(f) else float("-inf")
        ind.fitness = cache[key]
        return ind

    lo = MIN_HEIGHT[START]  # shallowest complete derivation
    depths = np.linspace(lo, max(lo, config.max_init_depth), config.population).round().astype(int)
    pop = [score(Individual(pi_grow(rng, int(d)))) for d in depths]
    best = max(pop, key=lambda i: i.fitness)
    history = [best.fitness]

    def tournament() -> Individual:
        idx = rng.choice(len(pop), size=min(config.tournament, len(pop)), replace=False)
        return max((pop[i] for i in idx), key=lambda i: i.fitness)

    for _ in range(config.generations):
        ranked = sorted(pop, key=lambda i: i.fitness, reverse=True)
        nxt = [Individual(i.tree.copy(), i.fitness) for i in ranked[: config.elitism]]
        while len(nxt) < config.population:
            a, b = tournament().tree, tournament().tree
            if rng.random() < config.crossover_prob:
                a, b = subtree_crossover(rng, a, b, config.max_depth)
            for child in (a, b):
                if rng.random() < config.mutation_prob:
                    child = subtree_mutation(rng, child, config.max_depth)
                if len(nxt) < config.population:
                    nxt.append(score(Individual(child.copy())))
        pop = nxt
        gen_best = max(pop, key=lambda i: i.fitness)
        if gen_best.fitness > best.fitness:
            best = Individual(gen_best.tree.copy(), gen_best.fitness)
        history.append(best.fitness)
    return RunResult(best, history, len(cache))
