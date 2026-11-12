import numpy as np
import pytest
import scipy.sparse as sp

from ge_tctn import measures
from ge_tctn.arff import load_arff
from ge_tctn.ge import GEConfig, evolve, pi_grow, subtree_crossover, subtree_mutation
from ge_tctn.grammar import GRAMMAR, MIN_HEIGHT, evaluate, map_genome
from ge_tctn.network import cooccurrence_stats, knn_graph, select_terms
from ge_tctn.protocol import Task, balanced_split, grid_tctn
from ge_tctn.tctn import initial_term_labels, llgc, weighted_f1, weighted_mean_f1


def random_docs(seed=0, n_docs=40, n_terms=12):
    rng = np.random.default_rng(seed)
    return sp.csr_matrix(rng.poisson(0.6, size=(n_docs, n_terms)).astype(float))


# --- grammar ---------------------------------------------------------------

def test_grammar_matches_thesis_figure_15():
    assert [len(GRAMMAR[s]) for s in ("<start>", "<sc>", "<op1>", "<op2>", "<t1>", "<t2>", "<n>")] \
        == [1, 7, 5, 5, 4, 4, 10]
    assert MIN_HEIGHT["<start>"] == 3


def test_piatetsky_shapiro_derivation_from_thesis_figure_16():
    # Thesis choices 0,0,6,0,1,4,0,2; the first 0 is <start>, which has a single option.
    tree = map_genome([0, 6, 0, 1, 4, 0, 2])
    assert tree.phenotype() == "((p(ti,tj))-((p(ti)*p(tj))))"
    stats = cooccurrence_stats(random_docs())
    np.testing.assert_allclose(evaluate(tree, stats), measures.piatetsky_shapiro(stats))
    assert map_genome(tree.genome()).phenotype() == tree.phenotype()


def test_mapping_fails_without_wraps_and_succeeds_with_them():
    assert map_genome([0]) is None
    assert map_genome([0], max_wraps=5) is None  # <sc> -> <sc><op1><sc> forever
    assert map_genome([6, 3], max_wraps=0).phenotype() == "p(~ti,~tj)"


def test_numbers_and_power():
    # <sc>(1): ((<sc>)<op1>(<n><n>.<n><n>)) with <sc>=p(ti,tj), op '^', number 02.00
    tree = map_genome([1, 6, 0, 4, 0, 2, 0, 0])
    assert tree.phenotype() == "((p(ti,tj))^(02.00))"
    stats = cooccurrence_stats(random_docs())
    np.testing.assert_allclose(evaluate(tree, stats), stats["pab"] ** 2)


def test_invalid_values_become_nan():
    tree = map_genome([3, 3, 0, 6, 0, 1, 4, 0, 2])  # ln of Piatetsky-Shapiro: NaN where < 0
    assert tree.phenotype() == "ln(((p(ti,tj))-((p(ti)*p(tj)))))"
    M = evaluate(tree, cooccurrence_stats(random_docs()))
    assert np.isnan(M).any() and np.all(np.isfinite(M[~np.isnan(M)]))


# --- network and measures ----------------------------------------------------

def test_cooccurrence_matches_brute_force():
    X = random_docs()
    s = cooccurrence_stats(X)
    B = X.toarray() > 0
    i, j = 2, 5
    assert s["pab"][i, j] == pytest.approx(np.mean(B[:, i] & B[:, j]))
    assert s["paB"][i, j] == pytest.approx(np.mean(B[:, i] & ~B[:, j]))
    assert s["pAb"][i, j] == pytest.approx(np.mean(~B[:, i] & B[:, j]))
    assert s["pAB"][i, j] == pytest.approx(np.mean(~B[:, i] & ~B[:, j]))
    assert (s["pa"] + s["pA"]).ravel() == pytest.approx(np.ones(12))


def test_measure_ranges():
    s = cooccurrence_stats(random_docs(seed=3))
    ps = measures.piatetsky_shapiro(s)
    assert ps.min() >= -0.25 and ps.max() <= 0.25
    for name in ("kappa", "yules_q"):
        v = measures.MEASURES[name](s)
        assert v.min() >= -1 - 1e-12 and v.max() <= 1 + 1e-12
    assert np.all(measures.mutual_information(s) >= -1e-12)
    assert np.allclose(measures.support(s), s["pab"])


def test_select_terms_keeps_top_tfidf():
    X = sp.csr_matrix(np.array([[5, 0, 1, 1], [5, 0, 1, 1], [0, 3, 1, 1]], dtype=float))
    # term 0 and 1 have df < n_docs; terms 2 and 3 occur everywhere (idf 0)
    assert list(select_terms(X, 0.5)) == [0, 1]


def test_knn_graph_symmetric_and_mutual():
    M = np.array([[0, 3, 1, 0], [3, 0, 2, 0], [1, 2, 0, 5], [0, 0, 5, 0]], dtype=float)
    Ws = knn_graph(M, 1, "symmetric").toarray()
    Wm = knn_graph(M, 1, "mutual").toarray()
    assert np.allclose(Ws, Ws.T) and np.allclose(Wm, Wm.T)
    assert Ws[0, 1] == 3 and Ws[2, 3] == 5 and Ws[1, 2] == 0
    assert Wm[0, 1] == 3 and Wm[2, 3] == 5 and Wm.sum() == 16


def test_knn_graph_breaks_ties_by_lowest_index():
    M = np.ones((5, 5))
    W = knn_graph(M, 2, "mutual").toarray()
    # each row picks the two lowest other indices: 0->{1,2}, 1->{0,2}, 2->{0,1}, 3->{0,1}, 4->{0,1}
    assert (W > 0).sum() == 6 and W[0, 1] and W[0, 2] and W[1, 2]
    # values differing only in the last bits count as ties
    M2 = M + np.random.default_rng(0).normal(0, 1e-15, M.shape)
    assert np.array_equal(knn_graph(M2, 2, "mutual").toarray() > 0, W > 0)


def test_knn_graph_ignores_nan_and_clips_negative():
    M = np.array([[0, np.nan, -1.0], [np.nan, 0, -2.0], [-1.0, -2.0, 0]])
    W = knn_graph(M, 1).toarray()
    assert np.all(W == 0)


# --- TCTN ----------------------------------------------------------------------

def test_initial_labels_eq_2_7():
    X = sp.csr_matrix(np.array([[2, 1, 0], [0, 1, 0]], dtype=float))
    Y = initial_term_labels(X, np.array([0, 1]), 2)
    assert np.allclose(Y, [[1, 0], [0.5, 0.5], [0, 0]])


def test_llgc_closed_form_equals_iteration():
    rng = np.random.default_rng(1)
    A = rng.random((8, 8))
    W = sp.csr_matrix(np.triu(A, 1) + np.triu(A, 1).T)
    Y = rng.random((8, 3))
    F = llgc(W, Y, 0.7)
    d = np.asarray(W.sum(1)).ravel()
    S = W.toarray() / np.sqrt(np.outer(d, d))
    G = Y.copy()
    for _ in range(500):
        G = 0.7 * S @ G + 0.3 * Y
    assert np.allclose(F, G)


def test_weighted_f1_variants():
    y, p = np.array([0, 0, 0, 1]), np.array([0, 0, 1, 1])
    assert weighted_f1(y, y, 2) == 1.0
    prec_w = 0.75 * 1.0 + 0.25 * 0.5
    rec_w = 0.75 * (2 / 3) + 0.25 * 1.0
    assert weighted_f1(y, p, 2) == pytest.approx(2 * prec_w * rec_w / (prec_w + rec_w))
    assert weighted_mean_f1(y, p, 2) == pytest.approx(0.75 * 0.8 + 0.25 * (2 / 3))


# --- GE ------------------------------------------------------------------------

def test_pi_grow_reaches_target_and_operators_keep_validity():
    rng = np.random.default_rng(0)
    for d in range(3, 11):
        t = pi_grow(rng, d)
        assert t.depth() <= d
        assert map_genome(t.genome()).phenotype() == t.phenotype()
    a, b = pi_grow(rng, 8), pi_grow(rng, 8)
    for _ in range(50):
        a, b = subtree_crossover(rng, a, b, 12)
        a = subtree_mutation(rng, a, 12)
        assert a.depth() <= 12 and b.depth() <= 12
        assert map_genome(a.genome()).phenotype() == a.phenotype()


def test_evolve_is_deterministic_and_improves():
    stats = cooccurrence_stats(random_docs(seed=2))
    target = measures.kappa(stats)

    def fitness(tree):
        M = evaluate(tree, stats)
        return -float(np.mean(np.abs(np.clip(np.nan_to_num(M, nan=10.0), -10, 10) - target)))

    cfg = GEConfig(population=20, generations=8, max_depth=12)
    r1, r2 = evolve(fitness, cfg, seed=5), evolve(fitness, cfg, seed=5)
    assert r1.best.phenotype == r2.best.phenotype and r1.history == r2.history
    assert r1.history[-1] >= r1.history[0]
    assert all(b >= a for a, b in zip(r1.history, r1.history[1:]))


# --- protocol and ARFF -----------------------------------------------------------

def write_arff(path, sparse=False):
    rng = np.random.default_rng(0)
    lines = ["@relation toy", ""]
    lines += [f"@attribute w{i} numeric" for i in range(30)]
    lines += ["@attribute class {a,b}", "@data"]
    for d in range(60):
        c = d % 2
        x = rng.poisson(0.3, 30)
        x[c * 15: c * 15 + 15] += rng.poisson(1.5, 15)
        if sparse:
            items = [f"{i} {v}" for i, v in enumerate(x) if v] + ([f"30 {'ab'[c]}"] if c else [])
            lines.append("{" + ",".join(items) + "}")
        else:
            lines.append(",".join(map(str, x)) + "," + "ab"[c])
    path.write_text("\n".join(lines) + "\n")
    return path


def test_arff_dense_and_sparse_agree(tmp_path):
    dense = load_arff(str(write_arff(tmp_path / "d.arff")))
    sparse = load_arff(str(write_arff(tmp_path / "s.arff", sparse=True)))
    assert dense.X.shape == (60, 30) and dense.classes == ["a", "b"]
    assert (dense.X != sparse.X).nnz == 0 and np.array_equal(dense.y, sparse.y)


def test_balanced_split():
    y = np.repeat([0, 1, 2], [50, 30, 20])
    s = balanced_split(y, 3, 0.1, np.random.default_rng(0))
    assert np.bincount(y[s.labelled]).tolist() == [3, 3, 3]
    assert np.bincount(y[s.validation]).tolist() == [3, 3, 3]
    assert len(np.intersect1d(s.labelled, s.test)) == 0 and len(s.test) == 82


def test_grid_tctn_learns_toy_problem(tmp_path):
    coll = load_arff(str(write_arff(tmp_path / "toy.arff")))
    task = Task(coll, term_fraction=1.0, n_splits=3, seed=0)
    res = grid_tctn(task, ks=(3,), alphas=(0.5,))
    assert len(res["table"]) == 5
    assert res["best"]["test"] > 0.8


def test_cli_smoke(tmp_path, capsys):
    from ge_tctn.cli import main
    path = write_arff(tmp_path / "toy.arff")
    out = tmp_path / "ge.json"
    assert main(["evolve", str(path), "--term-fraction", "1", "--splits", "2", "--k", "3",
                 "--alpha", "0.5", "--runs", "2", "--population", "6", "--generations", "2",
                 "--out", str(out)]) == 0
    assert "GE-TCTN" in capsys.readouterr().out and out.exists()
