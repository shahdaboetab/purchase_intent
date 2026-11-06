# Will This Visitor Buy? — Real-Time Purchase Intent

An e-commerce session-level classification project that estimates whether a visitor will purchase and recommends a response: **no action**, **show a coupon**, or **show a chat assistant**.

## Project deliverables

- **Exploration and analysis:** data quality checks, class-balance review, univariate/bivariate/multivariate EDA, and written observations in `purchase_intent.ipynb`.
- **Leakage-aware modeling:** stratified train/test split; preprocessing and feature engineering inside a scikit-learn pipeline; majority-class baseline; logistic regression, random forest, and gradient boosting; 5-fold stratified cross-validation and randomized hyperparameter search on training data.
- **Evaluation:** held-out test-set ROC-AUC, PR-AUC, precision/recall/F1 at a threshold selected from out-of-fold training predictions, confusion matrix, error analysis, and permutation importance. Results are printed in the notebook and summarized in the app.
- **Decision support:** a Streamlit session form returns purchase probability and an action. Low-intent sessions receive chat support, sessions above the tuned threshold but below the sure-buy cutoff may receive a coupon, and likely buyers receive no incentive.
- **Campaign stretch goal:** the notebook estimates extra sales from couponing the bottom 10% of predicted purchase probabilities. The assumed coupon uplift is clearly labeled as an assumption, not measured causal impact.

## Set up and run

Run commands from this project directory in PowerShell. Use a virtual environment so packages are not installed into a system-managed Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Train the pipeline and create `model.joblib` by opening `purchase_intent.ipynb` in VS Code with the Jupyter extension, selecting the `.venv` Python interpreter as the kernel, and choosing **Restart Kernel and Run All**. Keep this project directory as the notebook's working directory. The first run downloads the UCI dataset and requires internet access. It also writes `features.py`, evaluates the model, runs the campaign simulation, and saves the fitted artifact.

Then launch the app with Streamlit (do not run `app.py` directly with Python):

```powershell
streamlit run app.py
```

The app requires the generated `model.joblib` file in the same directory. It reports a clear setup message if that artifact is missing.

## Dataset and limitations

The [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468) contains approximately 12,000 sessions. The target is `Revenue` (`True` for a purchase). The target is imbalanced, so the project reports precision-recall metrics and tunes the decision threshold for the business objective rather than relying on accuracy alone.

`PageValues` can encode information from pages already viewed and may make a prediction more useful later in a session. The data represents one retailer and one year. The campaign estimate assumes a coupon uplift; actual incremental sales must be measured with a randomized A/B test.

## Repository contents

| File | Purpose |
|---|---|
| `purchase_intent.ipynb` | Data loading, EDA, modeling, evaluation, threshold tuning, campaign simulation, and artifact export |
| `features.py` | Shared session feature engineering used by the fitted pipeline |
| `app.py` | Streamlit prediction and action-recommendation app |
| `requirements.txt` | App, analysis, data-access, and notebook dependencies |
| `model.joblib` | Generated model artifact; created by running the notebook and not committed |
