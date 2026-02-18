# Barbell Exercise Tracking - Project Documentation

## 📋 Project Overview

This project implements a machine learning pipeline to classify and count repetitions of barbell exercises using accelerometer and gyroscope data from wearable sensors (MetaMotion devices). The system identifies five exercise types: bench press, squat, deadlift, overhead press (OHP), and barbell row.

---

## 🗂️ Project Structure

```
data-science-template/
├── data/
│   ├── raw/              # Original MetaMotion CSV files
│   ├── interim/          # Intermediate processed data (pickles)
│   └── processed/        # Final feature-engineered datasets
├── src/
│   ├── data/            
│   │   └── make_dataset.py          # Raw data loading & merging
│   ├── features/
│   │   ├── build_features.py        # Feature engineering pipeline
│   │   ├── count_repetitions.py     # Rep counting algorithm
│   │   ├── DataTransformation.py    # Filtering & PCA utilities
│   │   ├── TemporalAbstraction.py   # Time-domain features
│   │   ├── FrequencyAbstraction.py  # Frequency-domain features
│   │   └── remove_outliers.py       # Outlier detection
│   ├── models/
│   │   ├── train_model.py           # Model training & evaluation
│   │   ├── LearningAlgorithms.py    # ML algorithm wrappers
│   │   └── predict_model.py         # Prediction utilities
│   └── visualization/
│       ├── visualize.py             # Data visualization tools
│       └── plot_settings.py         # Matplotlib config
├── models/              # Saved trained models
├── reports/             # Analysis results & figures
├── environment.yml      # Conda environment specification
├── requirements.txt     # Pip dependencies
└── README.md           # Quick start guide
```

---

## 🔄 Data Processing Pipeline

### 1. **Data Ingestion** (`src/data/make_dataset.py`)
**Purpose:** Load and merge raw MetaMotion sensor data

**Key Operations:**
- Reads CSV files from `data/raw/MetaMotion/`
- Parses filenames to extract:
  - Participant ID (A-E)
  - Exercise label (bench, squat, dead, ohp, row)
  - Weight category (heavy, medium)
- Merges accelerometer & gyroscope data into unified DataFrame
- Saves to `data/interim/01_data_processed.pkl`

**Why:** Raw data comes in separate files per sensor/session. This consolidates everything into a single analyzable dataset.

---

### 2. **Outlier Removal** (`src/features/remove_outliers.py`)
**Purpose:** Detect and remove statistical anomalies

**Key Functions:**
- **Chauvenet's Criterion:** Removes points beyond mean ± k·std (probabilistic threshold)
- **IQR Method:** Filters points outside Q1-1.5·IQR to Q3+1.5·IQR
- **Local Outlier Factor (LOF):** Uses scikit-learn to detect density-based anomalies

**Output:** `data/interim/02_outliers_removed_chauvenets.pkl`

**Why:** Sensor noise and movement artifacts can skew features and harm model performance.

---

### 3. **Feature Engineering** (`src/features/build_features.py`)
**Purpose:** Transform raw sensor signals into meaningful ML features

#### 3.1 Missing Value Imputation
- Uses linear interpolation to fill NaN values in accelerometer/gyroscope columns

#### 3.2 Butterworth Low-Pass Filter (`DataTransformation.py`)
- **Purpose:** Remove high-frequency noise while preserving exercise patterns
- **Parameters:** 
  - Sampling rate: 5 Hz (1000ms / 200ms)
  - Cutoff: 1.3 Hz
  - Order: 5
- **Why:** Human movements occur at low frequencies; filtering improves signal clarity

#### 3.3 Principal Component Analysis (PCA)
- **Purpose:** Reduce dimensionality from 6 sensor axes to 3 components
- **Explained Variance:** Captures ~85-90% of total variance
- **Why:** Decorrelates sensor axes and reduces multicollinearity

#### 3.4 Sum of Squares (Magnitude Features)
- **acc_r:** √(acc_x² + acc_y² + acc_z²)
- **gyr_r:** √(gyr_x² + gyr_y² + gyr_z²)
- **Why:** Magnitude features are orientation-invariant and capture overall movement intensity

#### 3.5 Temporal Abstraction (`TemporalAbstraction.py`)
- **Window Size:** 5 samples (1 second)
- **Aggregations:** Rolling mean, std for each sensor
- **Output:** 16 features (8 sensors × 2 statistics)
- **Why:** Captures movement patterns over time (e.g., acceleration during lift phases)

#### 3.6 Frequency Abstraction (`FrequencyAbstraction.py`)
- **Method:** Fast Fourier Transform (FFT) on 10-sample windows
- **Features:**
  - Dominant frequency (max_freq)
  - Weighted average frequency (freq_weighted)
  - Power Spectral Entropy (pse)
  - Amplitudes at specific Hz bins
- **Why:** Different exercises have unique frequency signatures (e.g., squat has lower frequency than bench)

#### 3.7 K-Means Clustering
- **Purpose:** Discover hidden movement patterns
- **Optimal K:** 5 (determined by elbow method)
- **Input:** acc_x, acc_y, acc_z
- **Why:** Clusters may correspond to exercise phases or participant styles

**Final Output:** `data/interim/03_data_features.pkl` (100+ features)

---

### 4. **Repetition Counting** (`src/features/count_repetitions.py`)
**Purpose:** Automatically count exercise reps from sensor data

**Algorithm:**
1. Apply low-pass filter (cutoff=0.4 Hz) to acceleration magnitude
2. Detect local maxima using `scipy.signal.argrelextrema`
3. Count peaks as repetitions

**Why:** Peak detection works because each rep has a characteristic acceleration cycle

---

## 🤖 Model Training Pipeline (`src/models/train_model.py`)

### Feature Selection
**Forward Selection with Decision Tree:**
- Iteratively adds features that maximize training accuracy
- Stops at 10 features to prevent overfitting
- Selected subset includes: `pca_1`, `duration`, frequency features, temporal means

### Model Comparison
Tests 5 algorithms via grid search:

| Model | Key Hyperparameters | Use Case |
|-------|---------------------|----------|
| **Random Forest** | n_estimators, max_depth | Best overall performer (handles non-linear patterns) |
| **Neural Network** | hidden_layers, learning_rate | Captures complex feature interactions |
| **K-Nearest Neighbors** | n_neighbors, weights | Simple, interpretable baseline |
| **Decision Tree** | max_depth, min_samples_split | Fast, visualizable |
| **Naive Bayes** | - | Probabilistic baseline |

### Evaluation Strategy
1. **Random Split (80/20):** Initial model comparison
2. **Participant-Based Split:** Hold out Participant A to test generalization
   - **Why:** Tests if model works on new users (real-world scenario)

### Performance Metrics
- **Accuracy:** Overall correctness
- **Confusion Matrix:** Per-class error analysis
- **Feature Importance:** (from Random Forest) identifies most predictive features

---

## 📊 Key Utilities

### `DataTransformation.py`
- **LowPassFilter:** Butterworth filter (scipy.signal.butter + filtfilt)
- **PrincipalComponentAnalysis:** sklearn.decomposition.PCA wrapper with normalization

### `LearningAlgorithms.py` (from ML for Quantified Self book)
Provides grid search wrappers for:
- `feedforward_neural_network()` → MLPClassifier
- `random_forest()` → RandomForestClassifier
- `k_nearest_neighbor()` → KNeighborsClassifier
- `decision_tree()` → DecisionTreeClassifier
- `naive_bayes()` → GaussianNB
- `forward_selection()` → Greedy feature selection

---

## 🔗 File Dependencies

```
make_dataset.py
    ↓
remove_outliers.py
    ↓
build_features.py  ←─── DataTransformation.py
    ↓                    TemporalAbstraction.py
    ↓                    FrequencyAbstraction.py
    ↓
train_model.py  ←─────── LearningAlgorithms.py
    ↓
count_repetitions.py
```

---

## 🎯 Core Design Decisions

### Why Pickle Files?
- Preserves pandas DataFrames with datetime indices
- Faster I/O than CSV for large numerical datasets
- Maintains data types (no parsing overhead)

### Why Multiple Feature Sets?
Testing subsets (basic → full) helps identify:
- Which features add predictive value
- Computational cost vs. accuracy tradeoffs
- Risk of overfitting with too many features

### Why Participant-Based Split?
Standard train/test splits leak data when the same person appears in both sets. Holding out a participant tests **generalization to new users**.

---

## 📈 Expected Results

- **Best Model:** Random Forest with feature_set_4 (~95% accuracy)
- **Top Features:** PCA components, frequency features, temporal means
- **Rep Counting:** 90%+ accuracy on bench press (lower on complex exercises like deadlift)

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| **pandas** | Data manipulation |
| **numpy** | Numerical operations |
| **scikit-learn** | ML models, PCA, clustering |
| **scipy** | Signal processing (filters, peak detection) |
| **matplotlib** | Visualization |
| **seaborn** | Statistical plots |

---

## 📚 References

- **Book:** *Machine Learning for the Quantified Self* by Mark Hoogendoorn & Burkhardt Funk
- **Data:** MetaMotion wearable sensor recordings
- **Methodology:** Time-series feature engineering + supervised classification

---

## 🔮 Future Improvements

1. **Real-time inference:** Deploy model on edge device
2. **Form correction:** Use sensor data to detect improper technique
3. **Automatic set detection:** Remove manual set labeling
4. **Multi-sensor fusion:** Combine wrist + ankle sensors
5. **Deep learning:** LSTM/CNN for raw time-series (skip feature engineering)

---

## 📧 Contact

For questions or suggestions about this project structure, please refer to the repository maintainer.
