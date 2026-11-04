"""Create an explainable churn-risk workbench using only NumPy.

Scores are decision support. A human must review context before contacting,
pricing, or otherwise treating a customer differently.
"""

from __future__ import annotations

import argparse
import csv
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
REPORT_FILE = OUTPUT / "churn_risk_report.html"
DATA_FILE = OUTPUT / "demo_customers.csv"
FEATURES = ("tenure_months", "support_tickets", "recent_logins", "monthly_spend", "annual_contract")
DISPLAY_NAMES = {
    "tenure_months": "short tenure",
    "support_tickets": "many support tickets",
    "recent_logins": "low recent usage",
    "monthly_spend": "low monthly spend",
    "annual_contract": "no annual contract",
}


@dataclass
class Model:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def make_demo_data(rows: int = 1000, seed: int = 7) -> list[dict[str, float | int]]:
    """Generate realistic-looking, non-personal demonstration data."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int]] = []
    for customer_id in range(1, rows + 1):
        tenure = int(rng.integers(1, 61))
        tickets = int(rng.poisson(2.1))
        logins = int(rng.poisson(6.5))
        spend = round(float(np.clip(rng.normal(72, 27), 12, 180)), 2)
        annual_contract = int(rng.random() < 0.38)
        logit = -0.65 - 0.042 * tenure + 0.38 * tickets - 0.19 * logins - 0.010 * spend - 0.72 * annual_contract
        churned = int(rng.random() < sigmoid(np.array([logit]))[0])
        records.append(
            {
                "customer_id": customer_id,
                "tenure_months": tenure,
                "support_tickets": tickets,
                "recent_logins": logins,
                "monthly_spend": spend,
                "annual_contract": annual_contract,
                "churned": churned,
            }
        )
    return records


def feature_matrix(records: list[dict[str, float | int]]) -> np.ndarray:
    return np.array([[float(record[name]) for name in FEATURES] for record in records], dtype=float)


def labels(records: list[dict[str, float | int]]) -> np.ndarray:
    return np.array([int(record["churned"]) for record in records], dtype=float)


def train_logistic_regression(x: np.ndarray, y: np.ndarray, iterations: int = 2200, learning_rate: float = 0.08) -> Model:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1
    normalised = (x - mean) / std
    weights = np.zeros(normalised.shape[1])
    bias = 0.0
    for _ in range(iterations):
        probability = sigmoid(normalised @ weights + bias)
        error = probability - y
        weights -= learning_rate * (normalised.T @ error / len(y))
        bias -= learning_rate * float(error.mean())
    return Model(mean=mean, std=std, weights=weights, bias=bias)


def probability(model: Model, x: np.ndarray) -> np.ndarray:
    return sigmoid(((x - model.mean) / model.std) @ model.weights + model.bias)


def metrics(model: Model, x: np.ndarray, y: np.ndarray, threshold: float = 0.10) -> dict[str, float]:
    predictions = probability(model, x) >= threshold
    actual = y.astype(bool)
    true_positive = int(np.sum(predictions & actual))
    false_positive = int(np.sum(predictions & ~actual))
    false_negative = int(np.sum(~predictions & actual))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    accuracy = float(np.mean(predictions == actual))
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def reasons(model: Model, record: dict[str, float | int]) -> list[str]:
    values = np.array([float(record[name]) for name in FEATURES])
    contributions = ((values - model.mean) / model.std) * model.weights
    ranked = sorted(zip(FEATURES, contributions), key=lambda item: item[1], reverse=True)
    return [DISPLAY_NAMES[name] for name, contribution in ranked if contribution > 0][:2] or ["mixed customer signals"]


def recommended_action(risk: float, reason_list: list[str]) -> str:
    if risk >= 0.25:
        return f"Priority call within 24 hours; discuss {reason_list[0]}."
    if risk >= 0.10:
        return f"Send a helpful check-in and review {reason_list[0]}."
    return "No immediate action; continue normal service and monitor feedback."


def save_data(records: list[dict[str, float | int]]) -> None:
    with DATA_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def create_report(records: list[dict[str, float | int]], model: Model, result_metrics: dict[str, float]) -> None:
    x = feature_matrix(records)
    risks = probability(model, x)
    prioritised = sorted(zip(records, risks), key=lambda item: item[1], reverse=True)[:20]
    rows = []
    for record, risk in prioritised:
        reason_list = reasons(model, record)
        rows.append(
            "<tr>"
            f"<td>Customer {record['customer_id']}</td><td>{risk:.0%}</td><td>{html.escape(', '.join(reason_list))}</td>"
            f"<td>{html.escape(recommended_action(float(risk), reason_list))}</td></tr>"
        )
    metric_cards = "".join(
        f'<div class="card"><div class="muted">Test {name.title()}</div><div class="metric">{value:.0%}</div></div>'
        for name, value in result_metrics.items()
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Customer Churn Risk Workbench</title><style>
body{{font-family:Arial,sans-serif;background:#f8fafc;color:#172033;margin:0;padding:32px;line-height:1.5}}main{{max-width:1100px;margin:auto}}h1{{margin-bottom:4px}}.muted{{color:#5b6472}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin:24px 0}}.card,section{{background:white;border-radius:12px;padding:18px;box-shadow:0 1px 4px #cbd5e155}}.metric{{font-size:1.8rem;font-weight:bold;color:#b91c1c}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}}th{{background:#fef2f2}}.notice{{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px;border-radius:4px}}
</style></head><body><main>
<h1>Customer Churn Risk Workbench</h1><p class="muted">Generated data only - the tool demonstrates prioritisation, not automatic customer decisions.</p>
<div class="cards">{metric_cards}<div class="card"><div class="muted">Customers scored</div><div class="metric">{len(records)}</div></div></div>
<section class="notice"><strong>Human review required.</strong> Check the account context and customer preferences before contacting anyone. Do not use sensitive attributes in the model.</section>
<section><h2>Priority review queue</h2><p>Risk is a screening signal. Scores at or above 10% enter review, which favours recall; “why” makes the next step inspectable.</p><table><thead><tr><th>Account</th><th>Risk</th><th>Main signals</th><th>Suggested next step</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
</main></body></html>"""
    REPORT_FILE.write_text(page, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an explainable churn-risk workbench.")
    parser.add_argument("--score", action="store_true", help="Print a risk score for the values below.")
    parser.add_argument("--tenure", type=int, default=12)
    parser.add_argument("--tickets", type=int, default=2)
    parser.add_argument("--logins", type=int, default=5)
    parser.add_argument("--spend", type=float, default=70.0)
    parser.add_argument("--annual-contract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(exist_ok=True)
    records = make_demo_data()
    split = int(len(records) * 0.8)
    model = train_logistic_regression(feature_matrix(records[:split]), labels(records[:split]))
    result_metrics = metrics(model, feature_matrix(records[split:]), labels(records[split:]))
    save_data(records)
    create_report(records, model, result_metrics)
    print(f"Report created: {REPORT_FILE}")
    print("Test metrics: " + ", ".join(f"{name}={value:.0%}" for name, value in result_metrics.items()))
    if args.score:
        customer = {
            "tenure_months": args.tenure,
            "support_tickets": args.tickets,
            "recent_logins": args.logins,
            "monthly_spend": args.spend,
            "annual_contract": int(args.annual_contract),
        }
        risk = float(probability(model, feature_matrix([customer]))[0])
        reason_list = reasons(model, customer)
        print(f"Risk: {risk:.0%}; signals: {', '.join(reason_list)}")
        print("Suggested action: " + recommended_action(risk, reason_list))


if __name__ == "__main__":
    main()


