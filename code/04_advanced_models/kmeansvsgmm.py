# KNN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from scipy.stats import chi2_contingency
df = pd.read_csv("Shanghai_T2DM_Standardized.csv")
cluster_features = [
    # 人口學
    "z_Age (years)",
    "z_BMI (kg/m2)",
    "z_Duration of diabetes (years)",
    # 血糖控制
    "z_log_HbA1c (mmol/mol)",
    "z_log_Fasting Plasma Glucose (mg/dl)",
    # 血脂
    "z_log_Triglyceride (mmol/L)",
    "z_log_High-Density Lipoprotein Cholesterol (mmol/L)",
    "z_log_Low-Density Lipoprotein Cholesterol (mmol/L)",
    # 腎功能
    "z_log_Creatinine (umol/L)",
    "z_log_Estimated Glomerular Filtration Rate  (ml/min/1.73m2) "
]
X = df[cluster_features]

scores = []
#選最佳 K
#===============PCA
# 1. PCA降到2維，方便畫圖
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
#==============
kmeans = KMeans(
    n_clusters=2,
    random_state=42
)
df["KMeans_Cluster"] = kmeans.fit_predict(X)

kmeans_summary = df.groupby("KMeans_Cluster")[
    [
        "Age (years)",
        "BMI (kg/m2)",
        "HbA1c (mmol/mol)",
        "Duration of diabetes (years)"
    ]
].mean()

print("K-means Cluster Summary")
print(kmeans_summary)

print("\nK-means Cluster Counts")
print(df["KMeans_Cluster"].value_counts())

print("\nK-means Microvascular Proportion")
print(
    pd.crosstab(
        df["KMeans_Cluster"],
        df["Microvascular_Complication"],
        normalize="index"
    ) * 100
)

kmeans_score = silhouette_score(
    X,
    df["KMeans_Cluster"]
)

print("\nK-means Silhouette =", kmeans_score)

table = pd.crosstab(
    df["KMeans_Cluster"],
    df["Microvascular_Complication"]
)

chi2, p, dof, expected = chi2_contingency(table)

print("K-means Microvascular p-value =", p)

#正式分群
#==============

table = pd.crosstab(
    df["Cluster"],
    df["Microvascular_Complication"]
)

chi2,p,dof,expected = chi2_contingency(table)

print("p-value =",p)

#===========
#GMM
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,
    random_state=42
)

cluster = gmm.fit_predict(X)

df["GMM_Cluster"] = cluster
print(df["GMM_Cluster"].value_counts())
cluster_summary = df.groupby("GMM_Cluster")[
[
    "Age (years)",
    "BMI (kg/m2)",
    "HbA1c (mmol/mol)",
    "Duration of diabetes (years)"
]
].mean()

print(cluster_summary)

score = silhouette_score(
    X,
    df["GMM_Cluster"]
)

print("GMM Silhouette =", score)
#檢查併發症比例
pd.crosstab(
    df["GMM_Cluster"],
    df["Microvascular_Complication"],
    normalize="index"
) * 100
pd.crosstab(
    df["GMM_Cluster"],
    df["Macrovascular_Complication"],
    normalize="index"
) * 100
#=========卡方檢定
from scipy.stats import chi2_contingency

table = pd.crosstab(
    df["GMM_Cluster"],
    df["Microvascular_Complication"]
)

chi2,p,dof,expected = chi2_contingency(table)
print("Microvascular p-value =", p)

table = pd.crosstab(
    df["GMM_Cluster"],
    df["Macrovascular_Complication"]
)

chi2,p,dof,expected = chi2_contingency(table)
print("Macrovascular p-value =", p)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["KMeans_Cluster"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-means Clustering Result")
plt.tight_layout()
plt.show()

# =========================
# 5. GMM PCA圖
# =========================
plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["GMM_Cluster"]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("GMM Clustering Result")
plt.tight_layout()
plt.show()
