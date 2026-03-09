from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "clustering_model.pkl"

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
SEGMENT_LABELS = ["Economy", "Standard", "Premium", "Luxury", "Elite"]


def _remove_iqr_outliers(frame: pd.DataFrame, columns: list) -> pd.DataFrame:
    filtered = frame.copy()
    for col in columns:
        q1, q3 = filtered[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        filtered = filtered[
            (filtered[col] >= q1 - 1.5 * iqr) & (filtered[col] <= q3 + 1.5 * iqr)
        ]
    return filtered if not filtered.empty else frame


# ── Load & clean data ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
working_df = _remove_iqr_outliers(df.dropna(subset=SEGMENT_FEATURES).copy(), SEGMENT_FEATURES)
X_raw = working_df[SEGMENT_FEATURES].values

# ── Search for best k using a preprocessing pipeline ──────────────────────────
best_score = -1.0
best_k = 3
best_pipeline = None
best_labels = None

for k in range(2, 9):
    pipe = Pipeline([
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=500)),
    ])
    labels = pipe.fit_predict(X_raw)
    X_transformed = pipe[:-1].transform(X_raw)
    score = silhouette_score(X_transformed, labels)
    if score > best_score:
        best_score = score
        best_k = k
        best_pipeline = pipe
        best_labels = labels

clustering_pipeline = best_pipeline
working_df["cluster_id"] = best_labels

# ── Build label mapping sorted by mean income ─────────────────────────────────
kmeans_step = clustering_pipeline.named_steps["kmeans"]
centers_scaled = kmeans_step.cluster_centers_
sorted_by_income = centers_scaled[:, 0].argsort()
cluster_mapping = {
    int(sorted_by_income[i]): SEGMENT_LABELS[i]
    for i in range(best_k)
}

working_df["client_class"] = working_df["cluster_id"].map(cluster_mapping)
joblib.dump(clustering_pipeline, MODEL_PATH)

# ── Metrics ────────────────────────────────────────────────────────────────────
silhouette_avg = round(best_score, 4)

cv_income = round(
    working_df["estimated_income"].std() / working_df["estimated_income"].mean() * 100, 2
)
cv_price = round(
    working_df["selling_price"].std() / working_df["selling_price"].mean() * 100, 2
)

# ── Summaries ──────────────────────────────────────────────────────────────────
cluster_summary = (
    working_df.groupby("client_class")[SEGMENT_FEATURES]
    .mean()
    .reset_index()
    .rename(columns={
        "estimated_income": "avg_income",
        "selling_price": "avg_selling_price",
    })
)
cluster_counts = working_df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")

cv_by_cluster = (
    working_df.groupby("client_class")[SEGMENT_FEATURES]
    .agg(["mean", "std"])
    .reset_index()
)
cv_by_cluster.columns = [
    "client_class", "income_mean", "income_std", "price_mean", "price_std"
]
cv_by_cluster["CV Income (%)"] = (
    cv_by_cluster["income_std"] / cv_by_cluster["income_mean"] * 100
).round(2)
cv_by_cluster["CV Price (%)"] = (
    cv_by_cluster["price_std"] / cv_by_cluster["price_mean"] * 100
).round(2)

comparison_df = working_df[
    ["client_name", "estimated_income", "selling_price", "client_class"]
].copy()


def predict_cluster_label(income: float, price: float) -> str:
    cluster_id = int(clustering_pipeline.predict([[income, price]])[0])
    return cluster_mapping.get(cluster_id, f"Segment {cluster_id + 1}")


def evaluate_clustering_model():
    return {
        "silhouette": silhouette_avg,
        "best_k": best_k,
        "cv_income": cv_income,
        "cv_price": cv_price,
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "cv_table": cv_by_cluster[
            ["client_class", "CV Income (%)", "CV Price (%)"]
        ].to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }