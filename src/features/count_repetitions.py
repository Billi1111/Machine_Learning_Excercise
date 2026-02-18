from matplotlib import category
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = r"C:\Users\Bilal's Desktop\Desktop\Machine learning Excersise\data-science-template\src\features"

if script_dir not in sys.path:
    sys.path.append(script_dir)
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
df = pd.read_pickle(os.path.join(base_dir, "data", "interim", "01_data_processed.pkl"))

df = df[df["label"] != "rest"]
acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyro_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyro_r"] = np.sqrt(gyro_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
print("Unique labels:", df["label"].unique())
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]
# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df = bench_df
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyro_r"].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000 / 200
LowPass = LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
bench_set = (
    bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
    if not bench_df.empty
    else None
)
squat_set = (
    squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
    if not squat_df.empty
    else None
)
row_set = (
    row_df[row_df["set"] == row_df["set"].unique()[0]] if not row_df.empty else None
)
ohp_set = (
    ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]] if not ohp_df.empty else None
)
dead_set = (
    dead_df[dead_df["set"] == dead_df["set"].unique()[0]] if not dead_df.empty else None
)

bench_set["acc_r"].plot()
column = "acc_r"
LowPass.low_pass_filter(
    bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10
)[column + "_lowpass"].plot()


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
def count_repetitions(dataset, cutoff=0.4, order=10, column="acc_r"):

    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )

    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)[0]
    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(data[column + "_lowpass"])
    plt.scatter(peaks.index, peaks[column + "_lowpass"], marker="o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = data["label"].iloc[0].title()
    category = data["category"].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()

    return len(peaks)


count_repetitions(bench_set, cutoff=0.4)
count_repetitions(squat_set, cutoff=0.35)
count_repetitions(row_set, cutoff=0.65, column="gyro_x")
count_repetitions(ohp_set, cutoff=0.35)
count_repetitions(dead_set, cutoff=0.4)
# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)
reps_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()
reps_df["reps_pred"] = 0

for s in df["set"].unique():
    set_df = df[df["set"] == s]
    if not set_df.empty:
        exercise = set_df["label"].iloc[0]
        category = set_df["category"].iloc[0]
        if exercise == "bench":
            cutoff = 0.4
            column = "acc_r"
        elif exercise == "squat":
            cutoff = 0.35
            column = "acc_r"
        elif exercise == "row":
            cutoff = 0.65
            column = "gyr_x"
        elif exercise == "ohp":
            cutoff = 0.35
            column = "acc_r"
        elif exercise == "dead":
            cutoff = 0.4
            column = "acc_r"
        else:
            continue

        reps_df.loc[
            (reps_df["label"] == exercise)
            & (reps_df["category"] == category)
            & (reps_df["set"] == s),
            "reps_pred",
        ] = count_repetitions(set_df, cutoff=cutoff, column=column)
reps_df
# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
error = mean_absolute_error(reps_df["reps"], reps_df["reps_pred"])
reps_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()
print(f"Mean Absolute Error: {error}")
