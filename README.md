# Metric scoring code

This folder contains code for scoring the generated questions in the workbook.

## Files

- `semantic_extract.py` reads the workbook and separates generated questions from teacher questions.
- `semantic_metrics.swift` calculates Semantic Precision, Teacher Coverage, Semantic F1, and Diversity.
- `tfidf_cosine_reference.py` calculates global-corpus TF-IDF cosine similarity.

The semantic code uses Apple's built-in 300-dimensional English word embedding. It mean-pools the available word vectors for each question and L2-normalizes the result. Pairwise similarity is cosine similarity clamped to `[0, 1]`.

For generated-question vectors `G` and teacher-question vectors `T`:

- Semantic Precision: mean over `g in G` of `max(cosine(g, t) for t in T)`
- Teacher Coverage: mean over `t in T` of `max(cosine(g, t) for g in G)`
- Semantic F1: `2 * precision * coverage / (precision + coverage)`
- Diversity: `1 - mean(cosine(g_i, g_j))` for every unique generated-question pair

Run on macOS:

```bash
export AP_RESEARCH_WORKBOOK="/absolute/path/to/input-workbook.xlsx"
export METRIC_SWIFT_CACHE="/absolute/path/to/a/writable/cache-directory"
mkdir -p "$METRIC_SWIFT_CACHE"
python3 semantic_extract.py \
  | swift -module-cache-path "$METRIC_SWIFT_CACHE" semantic_metrics.swift \
  > semantic_scores.json
```

## TF-IDF cosine

Run it with:

```bash
export AP_RESEARCH_WORKBOOK="/absolute/path/to/input-workbook.xlsx"
python3 tfidf_cosine_reference.py > tfidf_cosine_scores.json
```

It requires Python and `openpyxl`.
