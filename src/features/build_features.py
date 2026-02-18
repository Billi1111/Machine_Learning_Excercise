import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --------------------------------------------------------------
# Fix import path (Works for both Interactive & Script mode)
# --------------------------------------------------------------
try:
    # This works if you run the whole file (Play button)
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # This works if you use Shift+Enter (Interactive Window)
    # We hardcode the path to your specific 'features' folder
    script_dir = r"C:\Users\Bilal's Desktop\Desktop\Machine learning Excersise\data-science-template\src\features"

if script_dir not in sys.path:
    sys.path.append(script_dir)

# Now these imports will work
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle(
    r"C:\Users\Bilal's Desktop\Desktop\Machine learning Excersise\data-science-template\data\interim\02_outliers_removed_chauvenets.pkl"
)
predictor_columns = list(df.columns[:6])

# plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
df.info()
for col in predictor_columns:
    df[col] = df[col].interpolate()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["set"] == 25]["acc_y"].plot()
df[df["set"] == 50]["acc_y"].plot()

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start
    df.loc[df["set"] == s, "duration"] = duration.seconds

duration_df = df.groupby("category")["duration"].mean()
duration_df.iloc[0] / 5
duration_df.iloc[1] / 10
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
LowPass = LowPassFilter()
fs = 1000 / 200
cutoff = 1.3
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

subset = df_lowpass[df_lowpass["set"] == 45]
print(subset["label"][0])

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]
# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pca_values)
plt.xlabel("Number of principal components")
plt.ylabel("Explained variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)
subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()
# --------------------------------------------------------------
# Sum of squares attributes
# -------------------------------------------------------------
df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2
df_squared["acc_r"] = acc_r
df_squared["gyr_r"] = gyr_r
subset = df_squared[df_squared["set"] == 14]
subset[["acc_r", "gyr_r"]].plot(subplots=True)
df_squared
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]

ws = int(1000 / 200)
df_temporal_labels = []
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, col, ws, "mean")
    df_temporal = NumAbs.abstract_numerical(df_temporal, col, ws, "std")

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()

    # 1. Create a NEW empty figure for this specific 'set'
    plt.figure(figsize=(12, 6))

    for col in predictor_columns:
        subset[col + "_temp_mean_ws_" + str(ws)].plot(label=col + " mean")
        subset[col + "_temp_std_ws_" + str(ws)].plot(label=col + " std")

    # 2. Add a title so you know which set this is
    plt.title(f"Set: {s}")
    plt.legend()

    # 3. Render this plot and CLEAR it from memory
    plt.show()
    plt.close()

    df_temporal_labels.append(subset)

pd.concat(df_temporal_labels).reset_index(drop=True)

subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot(subplots=True)
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot(subplots=True)


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
# ---------------------------------------------------------
# 1. SETUP & OPTIMIZATION
# ---------------------------------------------------------
# Reset index so rows start at 0 (Crucial for the math)
df_temporal = df_temporal.reset_index(drop=True)

# Create the copy
df_freq = df_temporal.copy()
FreqAbs = FourierTransformation()

# Define parameters
fs = int(1000 / 200)  # Sampling rate (you called this 'fd' before, stick to 'fs')
ws = int(2000 / 200)  # Window size

# OPTIMIZATION: Convert to float32 to save RAM
cols_to_float = df_freq.select_dtypes(include=["float64"]).columns
df_freq[cols_to_float] = df_freq[cols_to_float].astype("float32")

# ---------------------------------------------------------
# 2. THE "REAL" PROCESSING (The Instructor's Loop)
# ---------------------------------------------------------
# instead of running it on the whole dataset at once (which mixes sets),
# we loop through each set individually.
print("Processing frequencies per set...")

df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Processing Set {s}...")
    # Reset index for EACH set so the window starts at 0 for everyone
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()

    # Run the abstraction on ALL predictor columns
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

# Combine everything back together
df_freq = pd.concat(df_freq_list).reset_index(drop=True)
print("Processing Complete!")

# ---------------------------------------------------------
# 3. VERIFICATION PLOT
# ---------------------------------------------------------
# Now we plot to prove it worked
subset = df_freq[df_freq["set"] == 14].copy()

plot_cols = [
    "acc_y_max_freq",
    "acc_y_freq_weighted",
    "acc_y_pse",
    f"acc_y_freq_1.5_Hz_ws_{ws}",
    f"acc_y_freq_2.5_Hz_ws_{ws}",
]

subset[plot_cols].dropna().plot(
    figsize=(12, 6), marker="o", title="Final Frequency Features"
)
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()
# df_freq = df_freq.iloc[::2]

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
cluster_columns = ["acc_y", "acc_x", "acc_z"]

k_values = range(2, 10)
inertias = []
for k in k_values:
    subset = df_freq[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Sum of squared distances (Inertia)")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
df_cluster = df_freq.copy()
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection="3d")
for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Cluster {c}")
ax.set_xlabel("Acc X")
ax.set_ylabel("Acc Y")
ax.set_zlabel("Acc Z")
ax.legend()
plt.show()

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection="3d")
for c in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=f"Label {c}")
ax.set_xlabel("Acc X")
ax.set_ylabel("Acc Y")
ax.set_zlabel("Acc Z")
ax.legend()
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
interim_dir = os.path.join(base_dir, "data", "interim")
os.makedirs(interim_dir, exist_ok=True)
df_cluster.to_pickle(os.path.join(interim_dir, "03_data_features.pkl"))
