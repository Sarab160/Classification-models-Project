# 🚦 Network Traffic Classification Dashboard

A **Streamlit web app** that classifies network traffic into different types (e.g., normal or malicious) using multiple machine learning models. The dashboard allows users to view model accuracies and make real-time predictions by entering network traffic features.

---

## 📌 Features

- Train and evaluate multiple ML models:  
  - KNN, Logistic Regression, Decision Tree, GaussianNB, SVC (poly), Voting Classifier, Bagging (SVC), Random Forest  
- Compare model accuracies instantly  
- Make predictions with user-provided network traffic data  
- Interactive Streamlit dashboard for easy visualization  

---

## 📊 Dataset

- CSV file: `network_traffic.csv`  
- Features include:  
  - `Duration`, `SourcePort`, `DestinationPort`, `PacketCount`, `ByteCount`  
  - `SourceIP`, `DestinationIP`, `Protocol`  
- Target: `Label` (traffic class)  

---

## 🔧 Tech Stack

- **Python**  
- **Pandas**, **NumPy** — data handling  
- **Scikit-learn** — preprocessing, model training, ensemble methods  
- **Streamlit** — interactive dashboard  

---

## ⚙️ How to Use

1. Clone the repo:  
   ```bash
   git clone <repo-url>
   cd <repo-folder>
