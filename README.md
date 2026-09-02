# Student Placement & Salary Predictor

A two-stage machine learning system that predicts whether a student will be placed during campus recruitment, and — only if placement looks likely — estimates their starting salary.

**Live app:** https://placement-prediction-1-7zh4.onrender.com
**API:** https://placement-prediction-wxtg.onrender.com

> Note: hosted on Render's free tier, so the first request after inactivity may take 30–60s while the service spins up.

---

## Problem Statement

Given a student's academic record, skills, and recruitment signals (resume score, interview score, etc.), predict:

1. **Will this student be placed?** (classification)
2. **If placed, what starting salary can they expect?** (regression)

The two questions are deliberately handled as separate, sequential models rather than one combined model, because salary is only meaningful *conditional on* placement — a student who isn't placed has no salary to predict.

**Pipeline logic:**
\`\`\`
Raw student profile → Classifier → Placed / Not Placed
                                        │
                              if Placed → Regressor → Estimated salary
                              if Not Placed → salary not estimated
\`\`\`

---

## Dataset

~50,000 student records with academic, skills, and recruitment-outcome features (`Age`, `CGPA`, `Attendance_Percentage`, `Programming_Skill`, `Resume_Score`, `Interview_Score`, `Employability_Score`, `Placement_Status`, `Starting_Salary_USD`, and more).

---

## EDA — Key Findings

- **Target imbalance:** ~78% Placed vs 22% Not Placed. This shaped every modeling decision downstream (metric choice, resampling, class weighting).
- **Strongest predictors of placement** (by boxplot separation between Placed/Not Placed groups): `Interview_Score`, `Employability_Score`, `Resume_Score`, `CGPA`, `Academic_Performance`, `English_Proficiency`, `Leadership_Experience`.
- **Weak/no signal:** `Gender`, `University_Year`, `LinkedIn_Profile` — placement *rate* was roughly constant across these categories, so they contribute little.
- **Data leakage columns identified and excluded from features:** `Company_Tier`, `Career_Field`, `Placement_Mode`. These are outcomes of placement, not causes of it — a real "new" student wouldn't have these values available at prediction time.
- **`Starting_Salary_USD = 0` for unplaced students is not an outlier** — it's a valid category label ("no job"), so outlier handling (IQR clipping) was applied *only* to placed students' salaries, never to the full column.
- **Resume_Score** was heavily left-skewed (skew ≈ −2.2) and was treated with a `PowerTransformer` (Yeo-Johnson) before scaling; other numeric features were close enough to symmetric (|skew| < 0.6) to need only `StandardScaler`.

---

## Preprocessing

Built as a single `ColumnTransformer` combining five sub-pipelines, so the exact same transformation logic runs identically at training and inference time (no manual re-encoding in the API layer):

| Column type | Treatment |
|---|---|
| Skewed numeric (`Resume_Score`) | `PowerTransformer` (Yeo-Johnson) → `StandardScaler` |
| Other numeric | `StandardScaler` |
| Ordinal (`University_Year`, `Academic_Performance`, `English_Proficiency`) | `OrdinalEncoder` with explicit category order |
| Nominal (`Gender`, `Major`) | `OneHotEncoder` |
| Binary Yes/No | `OneHotEncoder(drop='if_binary')` |

---

## Modeling — Classification (Placement Status)

Baseline Random Forest showed the classic imbalanced-data failure mode: high overall accuracy (~82%) but poor recall on the minority "Not Placed" class (~32%) — the model was defaulting to the majority class whenever uncertain.

| Approach | Not Placed Recall | Not Placed F1 | Accuracy |
|---|---|---|---|
| Random Forest (plain) | 0.32 | 0.43 | 0.815 |
| Random Forest + `class_weight='balanced'` | 0.32 | 0.43 | 0.815 |
| Random Forest + SMOTE | 0.46 | 0.52 | 0.809 |
| XGBoost (full imbalance ratio as `scale_pos_weight`) | 0.65 | 0.54 | 0.750 |
| **XGBoost (`scale_pos_weight` tuned to 0.3)** | **0.64** | **0.54** | **0.758** |

**Final choice:** XGBoost with `scale_pos_weight=0.3`, selected by sweeping weight values and optimizing for F1 on the minority class rather than raw accuracy — because in this pipeline, misclassifying a "Not Placed" student as "Placed" propagates into a meaningless salary prediction downstream.

---

## Modeling — Regression (Starting Salary)

Trained **only on placed students** (using the classifier's *predicted* labels at inference time, not ground-truth labels — see note below).

Initial XGBoost run: **R² = 0.19**. Diagnosed rather than accepted:

1. Checked feature importances → one feature (`Employability_Score`) dominated at 27%; every other feature sat in a flat, near-random 2–3% band — a signature of weak available signal, not a modeling bug.
2. Sanity-tested by temporarily adding the excluded leakage columns (`Company_Tier`, `Career_Field`, `Placement_Mode`) back in. R² barely moved (0.19 → 0.21), which ruled out "missing feature" as the explanation and pointed to irreducible noise in the salary variable itself.
3. Tried log-transforming the (skewed) target — negligible change.
4. Ran `RandomizedSearchCV` over `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree` — improved to **R² = 0.266, MAE ≈ $9,871**.

**Takeaway documented rather than hidden:** salary in this dataset is only weakly explained by the available academic/skills features, even in the best case. This was treated as a dataset property to report, not a failure to fix by overfitting.

---

## Avoiding a Subtle Evaluation Bug

An early design question: when evaluating the salary model, should it be tested against students who were **actually** placed, or against students the **classifier predicted** as placed?

Testing against ground-truth "Placed" rows gives an artificially clean evaluation. In production, the salary model only ever sees whatever the classifier hands it — including the classifier's false positives. Evaluating end-to-end (classifier's predictions feeding the regressor) gives a realistic estimate of real-world pipeline performance instead of an optimistic, disconnected one.

---

## Serving — FastAPI

Two serialized `scikit-learn` `Pipeline` objects (`classifier_pipeline.pkl`, `salary_pipeline.pkl`) are loaded once at startup. A single `/predict` endpoint runs them in sequence:

\`\`\`python
placement = classifier_pipeline.predict(input_df)
if placement == "Placed":
    salary = salary_pipeline.predict(input_df)
else:
    salary = 0  # not estimated
\`\`\`

Input is validated with a Pydantic schema using `Literal` types and `Field` constraints (e.g. `CGPA` bounded to 0–4, `Gender` restricted to known categories) so malformed requests are rejected before they ever reach the model.

---

## Tech Stack

- **Modeling:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **API:** FastAPI, Pydantic, Uvicorn
- **Frontend:** HTML/CSS/JS (single-page form)
- **Deployment:** Render (backend + frontend)

---

## What I'd Improve With More Time

- Richer features for salary (location, negotiation-related signals) — the diagnostic work strongly suggests the ceiling is feature availability, not model choice.
- Threshold tuning on the classifier's predicted probabilities instead of the default 0.5 cutoff, for finer control over the precision/recall trade-off.
- Model monitoring / logging on the deployed API to track prediction distribution drift over time.
