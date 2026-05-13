# Paper2 Statistics Method

This note documents the v4.8 statistics pipeline contract. It is reference material only;
`.ai/ledger.json` remains the state source.

## Modes

- Manifest-only dry-run validates the planned experiment grid and writes `stats_plan.json`.
- Dry-run mode does not compute p-values, confidence intervals, bootstrap intervals, or effect sizes.
- Results-present mode requires real multi-seed metric records with `scenario`, `algorithm`, `seed`, and
  metric values.

## Seed Completeness

Within each scenario, all compared algorithms must share the same seed grid. Incomplete grids are rejected
unless the operator passes `--allow-partial`, in which case the report records the partial-grid issue.

## Estimates

For complete result records, the analyzer exports:

- Per algorithm/scenario mean, sample standard deviation, standard error, normal-approximation 95 percent
  confidence interval, and deterministic bootstrap confidence interval.
- Baseline-vs-comparison paired seed deltas where shared seeds exist.
- Cohen's d effect-size estimates against the configured baseline.

## Tests And Correction

When SciPy is available, paired baseline comparisons use Wilcoxon signed-rank tests. If SciPy is missing or
the sample is insufficient, the test is recorded as not run rather than silently fabricating a value.

Numeric pairwise p-values are corrected with Holm-Bonferroni across the generated report. The corrected value
and correction method are written to `pairwise_tests.csv` and `statistics_report.json`.

## Claim Boundary

L1 and L2 outputs are screening or candidate-validation evidence. Paper-level statistical claims require
L3 formal-verification runs and real multi-seed results; a dry-run manifest is never a result set.
