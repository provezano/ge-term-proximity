# ge-term-proximity

Grammatical evolution of proximity functions for term networks (GE-TCTN).

This repository is a reimplementation, written in 2026 from the text of my
master's thesis (ICMC-USP, 2019). The original code was lost, so nothing here
comes from it. Where the thesis does not say how something was done, the
choice made here is listed under "Deviations and assumptions" below.

## What the method does

TCTN (Rossi, Rezende and Lopes, 2015) classifies documents with very few
labelled examples. It builds a network whose vertices are terms, gives each
term a class distribution taken from the labelled documents, spreads those
distributions over the network with the LLGC label-propagation algorithm
(Zhou et al., 2004), and labels each unlabelled document with the class that
gets the highest frequency-weighted sum over its terms.

The edges of the term network are weighted by a proximity function computed
from document co-occurrence probabilities: p(ti,tj), p(ti,¬tj), p(¬ti,tj),
p(¬ti,¬tj) and the marginals p(ti), p(¬ti), p(tj), p(¬tj). TCTN leaves the
choice of that function to the user. The usual candidates are
Piatetsky-Shapiro, mutual information, kappa, Yule's Q and support, and which
one works best changes from one collection to another.

GE-TCTN builds the proximity function automatically. A context-free grammar
(thesis Figure 15, printed by `ge-tctn grammar`) generates expressions over
the eight probabilities, the operators + - * / ^, the functions sin, tanh,
sqrt, ln and log2, and constants of the form dd.dd. Grammatical evolution
searches that space. The fitness of a function is the weighted F1 that TCTN
reaches with it on a small validation sample of the collection. The best
function found is then used to classify the rest of the collection.

The code has five parts, all in `src/ge_tctn/`:

- `network.py`: term selection (top 10% of terms by summed TF-IDF),
  co-occurrence probabilities from one sparse product of the binary
  document-term matrix, and symmetric or mutual k-NN term graphs;
- `measures.py`: the five classic measures of thesis Table 4;
- `grammar.py`: the grammar, derivation trees, the standard GE
  genotype-to-phenotype mapping (with optional wrapping) and the vectorised
  evaluation of an evolved function on all term pairs at once;
- `ge.py`: a small GE engine on derivation trees (position-independent grow
  initialisation, subtree crossover, subtree mutation, tournament selection,
  generational replacement with elitism), seeded and deterministic;
- `tctn.py` and `protocol.py`: TCTN, LLGC (solved exactly with a sparse LU
  factorisation), the weighted F1 of the thesis, the sampling protocol,
  Grid-TCTN and GE-TCTN.

## Install and run

Python 3.10 or later.

```bash
pip install .            # or: pip install -e ".[test]" for development
pytest                   # 19 tests, about 2 seconds
```

The input is a bag-of-words ARFF file (dense or sparse rows) whose last
nominal attribute is the class. The collections of the thesis come from the
ICMC-USP LABIC repository; `validation/fetch_data.sh` downloads five of them
from the Internet Archive copy and checks their SHA-256 sums.

```bash
bash validation/fetch_data.sh data
ge-tctn info data/CSTR.arff
ge-tctn grammar
ge-tctn grid data/CSTR.arff --out grid_CSTR.json
ge-tctn evolve data/CSTR.arff --k 1,7 --alpha 0.1,0.5,0.9 \
    --runs 10 --population 50 --generations 30 --out ge_CSTR.json
```

`grid` runs Grid-TCTN: every combination of the five measures, k in
{1, 7, 17, 37, 57} and alpha in {0.1, 0.3, 0.5, 0.7, 0.9}, chosen by mean
validation F1 over 10 random samples. `evolve` runs GE-TCTN. Its defaults are
the thesis settings (population 100, 100 generations, crossover 0.75,
mutation 0.10, tournament 5, elitism 2, 10 runs for each k and alpha), which
cost about 11 CPU seconds per run on CSTR and far more on larger collections,
so the options above reduce them. `--fitness-splits all` computes each
fitness on all ten validation samples instead of one (see item 9 below). Both
commands print a one-line summary and
write every intermediate number to the JSON file. Seeds are fixed (`--seed`,
default 0), so a run gives the same result every time on the same machine.

## Validation

The thesis evaluated 37 collections. Five of the small ones are still
available: CSTR, Tr11, Tr12, Tr23 and Oh15. Their sizes match thesis Table 5
(299, 414, 313, 204 and 913 documents; the term counts differ by one because
the class attribute is not counted here). All numbers are weighted F1 on the
test documents, averaged over ten random samples; thesis values come from
Tables 6 and 10. `validation/run.sh` reproduces everything below in about 18
minutes of CPU.

Grid-TCTN uses exactly the protocol of the thesis (five measures, 25 values of
k and alpha, ten samples):

| collection | thesis | this code | setting chosen here |
|---|---|---|---|
| CSTR | 0.7443 (sd 0.0336) | 0.7116 (sd 0.0430) | mutual information, k=1, alpha=0.7 |
| Tr23 | 0.4327 (sd 0.1108) | 0.4288 (sd 0.1399) | kappa, k=1, alpha=0.3 |
| Tr12 | 0.4023 (sd 0.2373) | 0.3254 (sd 0.2160) | kappa, k=1, alpha=0.3 |
| Tr11 | 0.2306 (sd 0.1164) | 0.3960 (sd 0.1594) | kappa, k=1, alpha=0.1 |
| Oh15 | 0.7194 (sd 0.0149) | 0.7170 (sd 0.0189) | mutual information, k=1, alpha=0.7 |

The standard deviations over samples are of the same size as in the thesis on
every collection. The means agree within about one standard error (the
thesis sd divided by the square root of 10) on Tr23, Tr12 and Oh15. CSTR is
0.033 lower, about 3 standard errors. Tr11 is 0.165 higher, about 4.5
standard errors; there the thesis reports Grid-TCTN far below its own GE-TCTN
mean, while here the five measures all give 0.39 to 0.40. The random samples differ from the original
ones, and items 2, 3 and 7 of the list below may differ from the original
code; any of these can move a mean on collections this small.

GE-TCTN was run with smaller budgets than in the thesis, which used population
100, 100 generations and ten runs for each of the 25 values of (k, alpha):

| collection | GE budget and fitness | thesis best / mean (sd) | best run by validation | best run by test | mean of runs (sd) |
|---|---|---|---|---|---|
| CSTR | pop 50, 30 gen, 15 (k, alpha), one sample | 0.7760 / 0.7467 (0.0110) | 0.6628 | 0.7253 | 0.6592 (0.0534) |
| CSTR | pop 50, 30 gen, k=1 and 3 alphas, ten samples | 0.7760 / 0.7467 (0.0110) | 0.7440 | 0.7440 | 0.7147 (0.0220) |
| CSTR | pop 100, 100 gen, k=1, alpha=0.7, ten samples | 0.7760 / 0.7467 (0.0110) | 0.7224 | 0.7224 | 0.7134 (0.0074) |
| Oh15 | pop 50, 30 gen, 9 (k, alpha), one sample | 0.7233 / 0.7170 (0.0062) | 0.7067 | 0.7088 | 0.6930 (0.0144) |
| Tr23 | pop 30, 20 gen, k=1 and 3 alphas, one sample | 0.5305 / 0.5112 (0.0246) | 0.3346 | 0.4584 | 0.4087 (0.0694) |

The thesis result is not reproduced in size. In the thesis, GE-TCTN beats
Grid-TCTN on 35 of 37 collections, by 0.03 on CSTR (best run) and by 0.10 on
Tr23. Here, with the fitness computed on one sample per run, GE-TCTN is below
Grid-TCTN on all three collections tried, both for the run with the best
validation fitness and for the mean of the runs; only the best run judged on
test data is above it, on CSTR and Tr23. The cause is visible in the logs:
on CSTR the evolved functions reach a fitness of about 0.85 on the 28
validation documents of their own sample but only 0.70 on the validation
documents of all ten samples, so the search fits the noise of a very
small validation set. When the fitness is the mean over the ten samples,
which is how the thesis chooses the Grid-TCTN settings, GE-TCTN matches or
slightly beats Grid-TCTN on CSTR (mean 0.713 to 0.715, best 0.722 to 0.744,
against 0.712), the same direction as the thesis but a smaller gain. The
validation documents of one sample are test documents of another, so this
variant selects with slightly more information than a strict hold-out would
give; Grid-TCTN is selected the same way.

The likely reasons for the remaining gap are the reduced GE budgets, the
fitness question above, the reading of "best of 10 runs" (item 10), and the
choices listed below. Several evolved functions are worth a look: on Oh15 the
best function was p(¬ti,¬tj) - p(¬ti)p(¬tj), and on CSTR one run found
p(¬ti)p(tj) - p(¬ti,tj); both are algebraically equal to Piatetsky-Shapiro.

## Deviations and assumptions

The thesis describes the method and the protocol but not every detail of the
code. These are the places where this reimplementation had to choose.

1. Co-occurrence is counted at document level: p(ti,tj) is the share of all
   documents of the collection (labelled or not) that contain both terms,
   after term selection. The other cells of the contingency table follow from
   it and the marginals.
2. Term selection keeps the top 10% of terms by the sum over documents of
   tf x ln(N/df). The thesis says "sum of TF-IDF" without the exact variant.
3. The term graph is a symmetric k-NN graph: an edge joins i and j if j is
   among the k terms closest to i or the other way round, with the proximity
   value as weight. The thesis lists mutual k-NN, symmetric k-NN, epsilon
   neighbourhoods and the complete graph as options but does not say which one
   was used. `--graph mutual` switches to mutual k-NN.
4. Evolved functions can return negative, infinite or undefined values (for
   example ln of a negative number). Undefined and infinite values never become
   neighbours, and negative weights are set to zero, because LLGC needs
   non-negative weights. The thesis does not say how such values were handled.
5. LLGC uses S = D^-1/2 W D^-1/2 as in Zhou et al. (2004); the thesis prints
   D^1/2, which looks like a typo. The fixed point is computed exactly instead
   of by iteration; the result is the same up to the iteration tolerance.
6. A document gets the class with the largest score. The thesis says argmin,
   which would pick the least likely class, so argmax is assumed.
7. Weighted F1 follows thesis Eqs. 4.1 to 4.3: the harmonic mean of the
   support-weighted precision and the support-weighted recall. This is not the
   same number as scikit-learn's `f1_score(average="weighted")`, the
   support-weighted mean of the per-class F1, which is lower on these data
   (0.656 instead of 0.712 for Grid-TCTN on CSTR). The JSON output has both.
   Which one the original code computed is not known.
8. Samples: the labelled set and the validation set each take
   round(0.1 N / |C|) documents per class (fewer if a class is too small); the
   remaining documents, about 80%, are the test set. Ten samples are drawn from
   a fixed seed. Grid-TCTN and GE-TCTN choose their settings by mean
   validation F1 and report the mean test F1 over the same ten samples.
9. In GE-TCTN, run r of each (k, alpha) uses sample r for its fitness by
   default (`--fitness-splits one`). The thesis can also be read as using the
   mean over the ten samples, as Grid-TCTN does; `--fitness-splits all` does
   that, at ten times the cost. The (k, alpha) pair with the best mean fitness
   over its runs is kept.
10. The thesis reports the "best individual of 10 runs". It does not say
    whether "best" was judged on validation or on test data. The table below
    shows both: the run with the best validation fitness, and the largest test
    score over the runs. Only the first is a fair estimate.
11. GE details the thesis leaves to PonyGE2 defaults: maximum initial tree
    depth 10 (ramped from the smallest possible depth, 3), maximum tree depth
    90, no wrapping. Rules with a single option consume no codon, as in the
    thesis' description of the mapping (Section 3.2). Evolution works on
    derivation trees, so the linear genome is only used for the mapping
    itself and its tests. Subtree crossover is applied to a pair of parents
    with probability 0.75 and subtree mutation to each child with probability
    0.10. Fitness values are cached by phenotype.
12. The power operator is the real power; a negative base with a
    non-integer exponent gives an undefined value (see item 4).

Not reimplemented: the EM and document-network LLGC baselines of thesis
Table 6, and the Friedman and Li tests.

## How to cite

Please cite the thesis:

> Coutinho, F. P. (2019). *Construção Automática de Funções de Proximidade
> para Redes de Termos usando Evolução Gramatical* [Automatic construction of
> proximity functions for term networks using grammatical evolution].
> Master's dissertation, Instituto de Ciências Matemáticas e de Computação,
> Universidade de São Paulo, São Carlos. Defended 2019-12-04.
> https://doi.org/10.11606/D.55.2020.tde-16032020-150627

```bibtex
@mastersthesis{coutinho2019getctn,
  author = {Felipe Provezano Coutinho},
  title  = {Constru{\c{c}}{\~a}o Autom{\'a}tica de Fun{\c{c}}{\~o}es de Proximidade
            para Redes de Termos usando Evolu{\c{c}}{\~a}o Gramatical},
  school = {Instituto de Ci{\^e}ncias Matem{\'a}ticas e de Computa{\c{c}}{\~a}o,
            Universidade de S{\~a}o Paulo},
  address = {S{\~a}o Carlos, Brazil},
  year   = {2019},
  doi    = {10.11606/D.55.2020.tde-16032020-150627},
  url    = {https://teses.usp.br/teses/disponiveis/55/55134/tde-16032020-150627/}
}
```

`CITATION.cff` has the same information in machine-readable form.

## References

- R. G. Rossi, S. O. Rezende and A. A. Lopes (2015). Term network approach for
  transductive classification. Computational Linguistics and Intelligent Text
  Processing (CICLing 2015), Springer.
- D. Zhou, O. Bousquet, T. N. Lal, J. Weston and B. Schölkopf (2004). Learning
  with local and global consistency. Advances in Neural Information Processing
  Systems 16, 321-328.
- C. Ryan, J. J. Collins and M. O'Neill (1998). Grammatical evolution:
  evolving programs for an arbitrary language. EuroGP 1998.
- D. Fagan, M. Fenton and M. O'Neill (2016). Exploring position independent
  initialisation in grammatical evolution. IEEE CEC 2016.
- R. G. Rossi, R. M. Marcacini and S. O. Rezende (2013). Benchmarking text
  collections for classification and clustering tasks. ICMC-USP Technical
  Report 395.

## License

MIT, see `LICENSE`.
