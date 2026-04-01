# RUL-Predictor: Lithium-ion Battery Remaining Useful Life Estimation

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

RUL-Predictor is a machine learning solution for estimating the **Remaining Useful Life (RUL)** of lithium-ion batteries. By analyzing battery degradation patterns across charge-discharge cycles, this project predicts battery health and lifespan to support predictive maintenance and resource planning.

**Key Applications:**
- Electric vehicle battery management
- Energy storage system monitoring
- Industrial equipment maintenance planning
- Battery recycling and replacement scheduling

## ⚡ Features

- ✅ End-to-end ML pipeline for battery RUL prediction
- ✅ Multi-dataset support with automatic data consolidation
- ✅ Comprehensive data preprocessing and feature extraction
- ✅ Regression-based degradation modeling
- ✅ Performance visualization and degradation trend analysis
- ✅ Modular, extensible architecture for advanced models

## 📋 Requirements

- Python 3.8 or higher
- Dependencies listed in requirements (see installation)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/NikNaut/RUL-Predictor.git
cd RUL-Predictor
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib
```

### Usage

Run the predictor with:
```bash
python main.py
```

The script will:
1. Load and preprocess battery datasets
2. Train the regression model
3. Generate RUL predictions
4. Display performance metrics and visualizations

## 📁 Project Structure

```
RUL-Predictor/
├── batt_data5.csv          # Battery dataset 1
├── batt_data6.csv          # Battery dataset 2
├── batt_data7.csv          # Battery dataset 3
├── main.py                 # Main pipeline script
└── README.md               # This file
```

## 🔧 How It Works

### 1. **Data Loading**
- Reads multiple battery cycle datasets
- Consolidates data for comprehensive training coverage

### 2. **Preprocessing**
- Handles missing and noisy values
- Extracts key battery features:
  - Voltage (V)
  - Current (A)
  - Charge/Discharge cycles
  - Temperature (if available)

### 3. **Model Training**
- Applies regression algorithms to learn degradation patterns
- Maps battery operating conditions → Remaining Useful Life

### 4. **Prediction & Evaluation**
- Generates RUL estimates for battery conditions
- Compares predicted vs. actual values
- Provides performance metrics

## 📊 Output

The predictor generates:
- **RUL Estimates**: Predicted cycles remaining for each battery
- **Performance Metrics**: Error measurements (MAE, RMSE, R²)
- **Visualizations**: Battery degradation curves and trend analysis

## 🛠️ Tech Stack

| Component | Purpose |
|-----------|---------|
| **Python** | Core language |
| **Pandas** | Data manipulation & analysis |
| **NumPy** | Numerical computing |
| **Scikit-learn** | Machine learning models |
| **Matplotlib** | Data visualization |

## 💡 Future Enhancements

- [ ] Deep learning models (LSTM, RNN) for improved time-series prediction
- [ ] Advanced feature engineering techniques
- [ ] Web dashboard for interactive monitoring
- [ ] Real-time battery data streaming support
- [ ] Multi-chemistry battery support (NCA, LFP, etc.)
- [ ] Uncertainty quantification and confidence intervals

## 📝 License

This project is open for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## 📬 Contact

For questions or collaboration, reach out via GitHub Issues or discussions.

---

**Note:** This predictor is intended for research and educational use. For production systems, validate predictions with domain experts and real-world testing.