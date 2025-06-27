import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Font settings (change if needed)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

# Paths
DATA_PATH = "../results/eval_metrics_extended.csv"
OUTPUT_DIR = "../results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load evaluation data
df = pd.read_csv(DATA_PATH)

# 1. Overall average metrics
def plot_overall_bar():
    metrics = ["Hit@1", "Hit@5", "NDCG@5", "ILD@5", "PerUserPersonalization", "Personalization"]
    means = df[metrics].mean()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(means.index, means.values)
    for bar in bars:
        y = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y + 0.01, f"{y:.2f}", ha='center')
    plt.title("Overall Mean Scores of Evaluation Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/overall_mean_metrics.png")
    plt.close()

# 2-5. Per-user metric bar chart
def plot_user_metric_bar(metric, title, ylabel, filename):
    plt.figure(figsize=(12, 6))
    plt.bar(df['user_id'].astype(str), df[metric])
    plt.title(title)
    plt.xlabel("User ID")
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png")
    plt.close()

# 6. NDCG@5 distribution curve
def plot_ndcg_distribution():
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["NDCG@5"], fill=True, bw_adjust=0.25)
    plt.title("NDCG@5 Score Distribution (KDE)")
    plt.xlabel("NDCG@5")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ndcg_kde.png")
    plt.close()

# 7. Personalization vs. Diversity scatter plot
def plot_personal_vs_ild():
    plt.figure(figsize=(8, 6))
    plt.scatter(df["PerUserPersonalization"], df["ILD@5"], alpha=0.6)
    plt.title("Personalization vs. Diversity (ILD@5)")
    plt.xlabel("Per-User Personalization")
    plt.ylabel("ILD@5")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/scatter_personal_vs_ild.png")
    plt.close()

# Run all plots
plot_overall_bar()
plot_user_metric_bar("PerUserPersonalization", "Per-User Personalization Scores", "Score", "per_user_personalization")
plot_user_metric_bar("NDCG@5", "Per-User NDCG@5 Ranking Quality", "NDCG@5", "per_user_ndcg")
plot_user_metric_bar("Hit@5", "Per-User Hit@5 Accuracy", "Hit@5", "per_user_hit5")
plot_user_metric_bar("ILD@5", "Per-User ILD@5 Diversity", "ILD@5", "per_user_ild")
plot_ndcg_distribution()
plot_personal_vs_ild()

print("✅ All evaluation plots saved to:", OUTPUT_DIR)
