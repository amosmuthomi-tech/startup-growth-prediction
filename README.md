# Startup Growth Prediction
### Classifying high-growth startups · iHub Research · 2022–2023

Random Forest + XGBoost ensemble that flags startups likely to reach high growth, built during my work at iHub Research for a regional innovation-mapping project. Reported on the real (unpublished) data: **85% accuracy, AUC-ROC 0.89**, with SMOTE on the training set only, GridSearchCV (5-fold) and SHAP for interpretation.

The real dataset is not published. `src/demo_model.py` runs the same pipeline on **synthetic data** so you can see the method end to end.

## Run the demo
```bash
pip install -r requirements.txt
python src/demo_model.py
```
Prints accuracy, precision, recall, AUC on a held-out test set, then mean |SHAP| per feature. Scores on synthetic data say nothing about the real study.

## Method
- Features: funding, team (technical co-founder, size), market (B2B/B2C, vertical), traction (revenue growth, partnerships)
- Minority class (~25%) handled with SMOTE applied **only** to the training split; evaluation stays on the original distribution
- Soft-voting ensemble of Random Forest and XGBoost; SHAP for per-feature attribution
- Automated Python validation cut data-cleaning time by 40% and dataset error rates by 25%

## Limitations
- Small sample (~450 startups) and a single point-in-time label limit generalisation
- Accuracy alone is a weak guide with a 25% minority class; look at precision and recall too

`Python` `scikit-learn` `XGBoost` `imbalanced-learn` `SHAP`
