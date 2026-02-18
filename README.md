# Barbell Exercise Tracking with Machine Learning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system that classifies barbell exercises and counts repetitions using accelerometer and gyroscope data from wearable sensors.

---

## 🎯 Project Goals

- **Exercise Classification:** Identify 5 exercises (bench press, squat, deadlift, OHP, row)
- **Rep Counting:** Automatically count repetitions using signal processing
- **Generalization:** Build models that work across different users

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Conda (recommended) or pip

### Option 1: Using Conda (Recommended)

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate tracking-barbell-exercises
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Prepare the Data

Place your MetaMotion CSV files in `data/raw/MetaMotion/` directory.

```bash
# Load and merge raw sensor data
python src/data/make_dataset.py
```

**Output:** `data/interim/01_data_processed.pkl`

### 2. Remove Outliers

```bash
# Clean the data using Chauvenet's criterion
python src/features/remove_outliers.py
```

**Output:** `data/interim/02_outliers_removed_chauvenets.pkl`

### 3. Build Features

```bash
# Engineer time & frequency domain features
python src/features/build_features.py
```

**Output:** `data/interim/03_data_features.pkl` (100+ features)

### 4. Train Models

```bash
# Train and evaluate multiple ML models
python src/models/train_model.py
```

**Outputs:**
- Model comparison plots
- Confusion matrices
- Feature importance rankings

### 5. Count Repetitions

```bash
# Run repetition counting algorithm
python src/features/count_repetitions.py
```

**Output:** Visualization of detected peaks per exercise set

---

## 📂 Project Structure

```
data-science-template/
│
├── data/
│   ├── raw/              # Original MetaMotion CSV files
│   ├── interim/          # Processed pickles (01, 02, 03)
│   └── processed/        # Final datasets
│
├── src/
│   ├── data/
│   │   └── make_dataset.py          # Data loading pipeline
│   ├── features/
│   │   ├── build_features.py        # Feature engineering
│   │   ├── count_repetitions.py     # Rep counting
│   │   ├── DataTransformation.py    # Filters & PCA
│   │   ├── TemporalAbstraction.py   # Time-domain features
│   │   ├── FrequencyAbstraction.py  # FFT features
│   │   └── remove_outliers.py       # Outlier detection
│   ├── models/
│   │   ├── train_model.py           # Model training
│   │   └── LearningAlgorithms.py    # ML wrappers
│   └── visualization/
│       └── visualize.py             # Plotting utilities
│
├── models/              # Saved .pkl models
├── reports/             # Generated analysis
├── environment.yml      # Conda environment
├── requirements.txt     # Python dependencies
├── PROJECT_DOCUMENTATION.md  # Detailed technical docs
└── README.md           # This file
```

---

## 🧪 Running in Interactive Mode

If you prefer Jupyter-style execution:

1. Open any `.py` file in VS Code
2. Select the Python interpreter (`tracking-barbell-exercises`)
3. Run cells using `# %%` markers with **Shift+Enter**

---

## 📊 Expected Results

### Classification Accuracy
- **Random Forest:** ~95% (best model)
- **Neural Network:** ~93%
- **K-Nearest Neighbors:** ~88%

### Rep Counting
- **Bench Press:** 90-95% accuracy
- **Squat:** 85-90% accuracy
- **Other exercises:** 80-85% accuracy

---

## 🛠️ Key Technologies

| Component | Library | Purpose |
|-----------|---------|---------|
| Data Processing | `pandas`, `numpy` | DataFrame manipulation |
| ML Models | `scikit-learn` | Classification algorithms |
| Signal Processing | `scipy` | Filters, FFT, peak detection |
| Visualization | `matplotlib`, `seaborn` | Plotting |
| Feature Engineering | Custom modules | Time/frequency features |

---

## 📖 Documentation

For detailed explanations of:
- **File relationships**
- **Feature engineering rationale**
- **Algorithm choices**
- **Design decisions**

See **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)**

---

## 🔧 Troubleshooting

### ImportError: No module named 'X'
```bash
# Reinstall dependencies
conda env update -f environment.yml --prune
```

### FileNotFoundError for pickle files
Run the pipeline in order:
1. `make_dataset.py`
2. `remove_outliers.py`
3. `build_features.py`
4. `train_model.py`

### ConvergenceWarning in Neural Network
This is expected during grid search. The warnings don't affect final results.

### "No kernel connected" in VS Code
Click the kernel selector (top-right) and choose `tracking-barbell-exercises` environment.

---

## 📝 Data Format

### Input CSV Structure
```
epoch,time,elapsed,x,y,z
1547473369165,14:22:49.165,0,0.123,-0.456,9.789
```

### Filename Convention
```
{Participant}-{Exercise}-{Category}_MetaWear_{Timestamp}_{Sensor}_{Frequency}.csv

Example: A-bench-heavy_MetaWear_2019-01-14T14.22.49.165_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv
```

---

## 🎓 Learning Resources

This project demonstrates:
- ✅ End-to-end ML pipeline design
- ✅ Time-series feature engineering
- ✅ Model selection & hyperparameter tuning
- ✅ Cross-validation strategies
- ✅ Signal processing techniques

**Based on:** *Machine Learning for the Quantified Self* by Hoogendoorn & Funk

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📧 Contact

For questions or issues:
- Open an issue in the repository
- Check `PROJECT_DOCUMENTATION.md` for technical details

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **Data Collection:** MetaMotion sensor platform
- **Methodology:** ML for Quantified Self book
- **Template:** Cookie Cutter Data Science

---

**Happy Training! 🏋️‍♂️**
