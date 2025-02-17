# **Advanced Fraud Detection with Anomaly Detection**

![GitHub stars](https://img.shields.io/github/stars/nafisalawalidris/Advanced-Fraud-Detection-with-Anomaly-Detection)
![GitHub license](https://img.shields.io/github/license/nafisalawalidris/Advanced-Fraud-Detection-with-Anomaly-Detection)
![GitHub issues](https://img.shields.io/github/issues/nafisalawalidris/Advanced-Fraud-Detection-with-Anomaly-Detection)

This project focuses on enhancing the sensitivity and accuracy of fraud detection systems by combining supervised learning and anomaly detection techniques. The goal is to identify both known and novel fraud patterns, ensuring that financial institutions can detect fraudulent activities more effectively and reduce losses.

### **If you find this project useful, please consider giving it a star ⭐ on GitHub. Contributions are also welcome!**

![alt text](<Advanced Fraud Detection with Anomaly Detection API.png>)

---

## **Objectives**

- **Improve Fraud Detection:** Enhance the existing fraud detection system by incorporating advanced anomaly detection techniques.
- **Identify Novel Fraud Patterns:** Use unsupervised learning methods to detect rare or previously unseen fraud patterns.
- **Minimise False Positives:** Ensure the system accurately identifies fraudulent transactions while minimising false alarms.
- **Deploy the Model:** Create a real-time fraud detection system using FastAPI for deployment.

---

## **Tools and Technologies**

### **Python Libraries**:
- **pandas, numpy:** Data manipulation and preprocessing.
- **scikit-learn:** Machine learning models and evaluation metrics.
- **tensorflow, keras:** Deep learning models (if required).
- **matplotlib, seaborn:** Data visualisation.
- **joblib:** Saving and loading trained models.
- **fastapi, uvicorn:** Deployment of the model as a REST API.

### **Version Control**:
- GitHub for collaboration and version control.

---

## **Dataset**

The dataset used in this project contains credit card transactions, with the following features:
- **V1-V28:** Anonymized features representing transaction details.
- **Amount:** The transaction amount.
- **Class:** The target variable (0 for non-fraudulent, 1 for fraudulent).

---

## **Key Steps in the Project**

### 1. Import Libraries and Data Collection
- Import the necessary libraries and load the dataset containing historical credit card transactions.
- Inspect and prepare the dataset for preprocessing.

### 2. Load and Preprocess Dataset
- Handle missing values.
- Encode categorical variables (if any).
- Normalize numerical features using `StandardScaler` or `MinMaxScaler`.
- Split the dataset into training and testing sets.

### 3. Model Selection
- Use a combination of traditional machine learning models and anomaly detection techniques:

#### Supervised Learning Models:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

#### Anomaly Detection Models:
- Isolation Forest
- One-Class SVM
- Autoencoder Model

### 4. Train Fraud Detection Model (Supervised Learning)
- Train supervised learning models using labeled data.
- Evaluate models for accuracy in classifying fraudulent transactions.

### 5. Train Anomaly Detection Model
- Train anomaly detection models to identify outliers or anomalies.
- Detect novel fraud patterns not represented in the training data.

### 6. Ensemble Learning
- Combine supervised and anomaly detection models using techniques like majority voting or weighted averaging.

### 7. Evaluation Metrics
Evaluate model performance using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### 8. Save Trained Models
- Serialise trained models using `joblib` or `pickle`.

### 9. Deployment
- Deploy the models using FastAPI to create a REST API for real-time predictions.

### 10. Test FastAPI Application
- Test the deployed API with sample transaction data.
- Monitor the application for performance or prediction errors.

---

## **Getting Started**

### **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/nafisalawalidris/Advanced-Fraud-Detection-with-Anomaly-Detection.git


## Installation

1. **Clone the Repository**
```bash
   git clone https://github.com/nafisalawalidris/Fraud-Detection-with-Supervised-Learning.git
```

2. Navigate to the project directory:
```bash
cd Advanced-Fraud-Detection-with-Anomaly-Detection
```

3. Create a virtual environment:
```bash
python -m venv Advanced-Fraud-Detection-env
```

4. Activate the virtual environment:
- On Windows:
```bash
.\Advanced-Fraud-Detection-env\Scripts\activate
```
- On macOS/Linux
```bash
source Advanced-Fraud-Detection-env/bin/activate
```

5. Install the required packages:
```bash
pip install -r requirements.txt
```

## **Contributing**
Contributions are welcome! If you have suggestions for improvements or want to contribute to this project, follow these steps:

### **How to Contribute**
```bash
1. Fork the repository.
2. Create a new feature branch (git checkout -b feature-name).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature-name).
5. Open a pull request.
```

## **License**
This project is licensed under the MIT License. See the LICENSE file for more information.

## **Contact**
For any inquiries or feedback, feel free to reach out: https://nafisalawalidris.github.io/13/.