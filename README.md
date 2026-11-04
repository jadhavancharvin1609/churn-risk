# Customer Churn Risk Workbench 

A customer-success tool that ranks accounts by churn risk and recommends a human follow-up. It trains a logistic-regression model implemented with NumPy, so the model logic remains visible and explainable.

## Skills demonstrated

- Feature engineering and train/test separation
- Logistic regression, precision, recall, and accuracy
- Per-customer explanations and an action queue
- A safety rule: the score supports human review; it does not make an automatic customer decision

## Run it

```powershell
python -m pip install -r requirements.txt
python app.py
```

This creates `output/churn_risk_report.html` and `output/demo_customers.csv`.

## Score one customer

```powershell
python app.py --score --tenure 2 --tickets 7 --logins 1 --spend 25
```

Add `--annual-contract` when the customer has an annual contract.

## Portfolio upgrade ideas

- Replace synthetic data with approved, labelled historical data.
- Add SHAP values, calibration checks, fairness checks, and drift monitoring.
- Test whether recommended retention actions actually reduce churn.
