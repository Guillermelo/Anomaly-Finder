# Anomaly-Finder
This project combines an autoencoder neural network with an Isolation Forest algorithm to detect anomalies in network session data. The data is derived from raw TCP logs and transformed into structured session-based features.


## Project Overview

The primary objective of this project is to identify anomalous behavior in network traffic using unsupervised machine learning. The architecture consists of two main components:

1. A **deep autoencoder** trained to learn low-dimensional representations of normal sessions, minimizing reconstruction error.
2. An **Isolation Forest** trained on the encoded data to further isolate outliers based on structural deviation from normal session behavior.

By combining these two methods, the system enhances anomaly detection accuracy by leveraging both reconstruction error and decision boundary learning.

## Raw Data Analysis

Prior to model training, raw **TCP log data** was collected and analyzed. These logs contained basic connection-level details, and required careful preprocessing to extract meaningful **session-based metadata**.

Key preprocessing steps included:
- Conversion of boolean and categorical fields into numeric values.
- One-hot encoding for protocol types.
- Aggregation of session-level statistics (e.g., packet count, duration, port diversity).
- Feature engineering for flags such as `is_scan_like`, `is_unusual_protocol`, and session direction/role.

This step was crucial to convert low-level TCP activity into structured data suitable for modeling and anomaly detection.

## Model Architecture

### 🔸 Autoencoder

A fully connected **autoencoder neural network** was trained using PyTorch. It learns a compressed encoding of each session and attempts to reconstruct the original input from this lower-dimensional space.

The reconstruction error (measured via Mean Squared Error) provides an initial anomaly signal — sessions with unusually high reconstruction errors are candidates for being anomalous.

### 🔸 Isolation Forest

An **Isolation Forest** (from scikit-learn) was trained on the encoded outputs of the autoencoder to explicitly detect outliers. It assigns each session a decision score and an anomaly label (`-1` for anomalous, `1` for normal), offering a second perspective beyond reconstruction error.

This dual-stage anomaly detection approach improves robustness and interpretability.

```python
from sklearn.ensemble import IsolationForest

iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iforest.fit(encoded_data)

# Predictions
iforest_labels = iforest.predict(encoded_data)      # -1 = anomaly, 1 = normal
iforest_scores = iforest.decision_function(encoded_data)
```

The final output includes:
- `reconstruction_error`
- `iforest_score`
- `iforest_label`

## Technologies Used

- Python
- PyTorch
- scikit-learn
- Pandas
- Jupyter Notebook

## Dataset

The dataset (`sessions_day_output.csv`) contains extracted features from raw TCP logs. Each row represents a network session enriched with engineered features and categorical encodings.

## How to Run

1. Install dependencies:
   ```bash
   pip install pandas torch scikit-learn
   ```

2. Open and execute the notebook:
   ```bash
   jupyter notebook autoencoder.ipynb
   ```

3. After training, the notebook outputs:
   - A plot or distribution of reconstruction errors.
   - Anomaly labels and scores from the Isolation Forest.
   - A combined DataFrame showing both error-based and structural anomalies.

## Results

Sessions flagged by both the autoencoder and the Isolation Forest are highly likely to be anomalous. These can be prioritized for further inspection in a security or monitoring pipeline.

This hybrid model architecture offers a flexible and powerful approach to **unsupervised anomaly detection in network traffic**.
