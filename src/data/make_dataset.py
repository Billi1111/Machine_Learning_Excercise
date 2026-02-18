# %%

import pandas as pd
from glob import glob


import os

# Move up two levels: from src/data to src, then to data-science-template
if os.getcwd().endswith("Machine learning Excersise"):
    os.chdir("data-science-template")

# If in the src/data folder, move up to the project root
elif os.getcwd().endswith("data"):
    os.chdir("../../")

print(f"Final Working Directory: {os.getcwd()}")
print(f"Working directory: {os.getcwd()}")
print(
    f"File exists: {os.path.exists('data/raw/MetaMotion/MetaMotion/A-bench-heavy2_MetaWear_2019-01-14T14.27.00.784_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')}"
)

# List what files are actually in the directory
import os

data_dir = "data/raw/MetaMotion/MetaMotion/"
if os.path.exists(data_dir):
    print(f"Files in {data_dir}:")
    print(os.listdir(data_dir)[:5])  # Show first 5 files
else:
    print(f"Directory does not exist: {data_dir}")

# %%
# Read single CSV file
single_file_acc = pd.read_csv(
    "data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
)
single_file_gyr = pd.read_csv(
    "data/raw/MetaMotion/MetaMotion/A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
)
# %%
# Transform the data

# %%
# View the data (This creates the pretty table)
single_file_acc
single_file_gyr
# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("data/raw/MetaMotion/MetaMotion/*.csv")
len(files)

# --------------------------------------------------------------
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

f = files[0]
data_path = "data/raw/MetaMotion/MetaMotion/"

# 1. Normalize the path slashes so replace() works every time
f_normalized = f.replace("\\", "/")

# 2. Isolate the filename by removing the directory path
filename_only = f_normalized.replace(data_path, "")

# 3. Use split to extract features
parts = filename_only.split("-")
participant = parts[0]
label = parts[1]
category = parts[2].split("_")[0]

# --------------------------------------------------------------
# Read file and assign PROPERLY named columns
# --------------------------------------------------------------
df = pd.read_csv(f)

df["participant"] = participant  # Use "quotes" to name the column
df["label"] = label
df["category"] = category

df.head()  # Check the pretty table
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    # 1. Normalize the path slashes so replace() works every time
    f_normalized = f.replace("\\", "/")

    # 2. Isolate the filename by removing the directory path
    filename_only = f_normalized.replace(data_path, "")

    # 3. Use split to extract features
    parts = filename_only.split("-")
    participant = parts[0]
    label = parts[1]

    # Determine category by checking if Accelerometer or Gyroscope is in filename
    if "Accelerometer" in filename_only:
        category = "Accelerometer"
    elif "Gyroscope" in filename_only:
        category = "Gyroscope"
    else:
        category = "unknown"

    # Read the CSV file
    temp_df = pd.read_csv(f)

    # Assign new columns
    temp_df["participant"] = participant
    temp_df["label"] = label
    temp_df["category"] = category
    print(category)

    # Append to the correct DataFrame based on category
    if category == "Accelerometer":
        temp_df["set"] = acc_set
        acc_df = pd.concat([acc_df, temp_df], ignore_index=True)
        acc_set += 1
    elif category == "Gyroscope":
        temp_df["set"] = gyr_set
        gyr_df = pd.concat([gyr_df, temp_df], ignore_index=True)
        gyr_set += 1

acc_df[acc_df["set"] == 10]
# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.info()
pd.to_datetime(df["epoch (ms)"], unit="ms")
# pd.to_datetime(df["time (01:00)"]).dt.month
acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
del acc_df["epoch (ms)"]
del acc_df["time (01:00)"]
del acc_df["elapsed (s)"]

del gyr_df["epoch (ms)"]
del gyr_df["time (01:00)"]
del gyr_df["elapsed (s)"]


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
files = glob("data/raw/MetaMotion/MetaMotion/*.csv")


def read_data_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        # 1. Normalize the path slashes so replace() works every time
        f_normalized = f.replace("\\", "/")

        # 2. Isolate the filename by removing the directory path
        filename_only = f_normalized.replace(data_path, "")

        # 3. Use split to extract features
        parts = filename_only.split("-")
        participant = parts[0]
        label = parts[1]
        category = parts[2].split("_")[0]  # Extract weight category (heavy, medium)

        # Determine sensor type by checking if Accelerometer or Gyroscope is in filename
        if "Accelerometer" in filename_only:
            sensor_type = "Accelerometer"
        elif "Gyroscope" in filename_only:
            sensor_type = "Gyroscope"
        else:
            sensor_type = "unknown"

        # Read the CSV file
        temp_df = pd.read_csv(f)

        # Assign new columns
        temp_df["participant"] = participant
        temp_df["label"] = label
        temp_df["category"] = category

        # Append to the correct DataFrame based on sensor type
        if sensor_type == "Accelerometer":
            temp_df["set"] = acc_set
            acc_df = pd.concat([acc_df, temp_df], ignore_index=True)
            acc_set += 1
        elif sensor_type == "Gyroscope":
            temp_df["set"] = gyr_set
            gyr_df = pd.concat([gyr_df, temp_df], ignore_index=True)
            gyr_set += 1

    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    return acc_df, gyr_df


acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

# = data_merged.dropna() drops na values
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz
sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

data_merged.columns
data_merged[:1000].resample(rule="200ms").mean()
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
days[-1]
data_resampled = pd.concat(
    [df.resample(rule="200ms").agg(sampling).dropna() for df in days]
)
data_resampled.info()
data_resampled["set"] = data_resampled["set"].astype("int")
# --------------------------------------------------------------
# Export dataset
data_resampled.to_pickle("data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# %%
