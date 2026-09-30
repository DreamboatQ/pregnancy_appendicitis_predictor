# Pregnancy-Associated Appendicitis Prediction Website

This local Flask application reproduces the input-to-result pattern shown in Figures 10–11 of the supplied eClinicalMedicine paper, while using the model selected in `paper.tex`.

## What the application computes

- 15 predictors in the exact order saved by the study pipeline.
- 20 matched `IterativeImputer` + `StandardScaler` + Random Forest pipelines.
- Positive-class probability pooled by the arithmetic mean across the 20 models.
- Patient-specific class-1 SHAP values calculated separately for every model and averaged across imputations.
- The prespecified classification threshold of 0.50.

The deployed Random Forest parameters are `n_estimators=50`, `max_depth=5`, `max_features="log2"`, `min_samples_split=2`, `min_samples_leaf=1`, and `random_state=42`.

## Run locally

The existing `xgboostbest-mac` Conda environment in this workspace is supported directly:

```bash
./run_local.sh
```

Then open <http://127.0.0.1:5050>.

For a clean environment:

```bash
conda env create -f environment.yml
conda run --no-capture-output -n pregnancy-appendicitis-web python app.py
```

## Important limitation

The output predicts the observed historical decision to perform surgery in the retrospective development cohort. It does **not** estimate whether surgery is necessary or beneficial, has not been externally validated, and is not a treatment recommendation or a screening/confirmatory test.
