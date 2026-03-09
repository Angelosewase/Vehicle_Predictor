import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "clustering_model.pkl"

SEGMENT_FEATURES = ["estimated_income", "selling_price"]
df = pd.read_csv(DATA_PATH)
X = df[SEGMENT_FEATURES]
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["cluster_id"] = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_
# Sort clusters by income
sorted_clusters = centers[:, 0].argsort()
cluster_mapping = {
 sorted_clusters[0]: "Economy",
 sorted_clusters[1]: "Standard",
 sorted_clusters[2]: "Premium",
}
df["client_class"] = df["cluster_id"].map(cluster_mapping)
joblib.dump(kmeans, MODEL_PATH)
silhouette_avg = round(silhouette_score(X, df["cluster_id"]), 2)
cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")
comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]
def evaluate_clustering_model():
 return {
 "silhouette": silhouette_avg,
 "summary": cluster_summary.to_html(
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