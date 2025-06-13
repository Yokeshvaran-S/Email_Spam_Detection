# 📧 Email Spam Detection using Logistic Regression and TF-IDF

This project is a **Machine Learning-based web application** built with **Streamlit** that detects whether an email is spam or not. It uses **Logistic Regression** trained on the **SpamAssassin dataset**, with **TF-IDF vectorization** for feature extraction.

---

## 💡 Overview

Spam emails are unsolicited messages that often contain malicious links, ads, or scams. This application helps detect such messages using Natural Language Processing (NLP) and a Logistic Regression classifier.

The project includes:

- Data cleaning and preprocessing
- TF-IDF vectorization of email text
- Logistic Regression model training
- Streamlit app with user-friendly interface
- Visualization of spam probability

---

## 🧠 Machine Learning Model

- **Model:** Logistic Regression
- **Feature Extraction:** TF-IDF Vectorizer with max 10,000 features
- **Training/Test Split:** 80/20 ratio
- **Evaluation Metrics:**
  - Accuracy
  - Classification Report (Precision, Recall, F1-score)

---

## 🗂 Project Structure

email-spam-detector/
│
├── spamassassin_raw.csv # Raw email dataset
├── app.py # Streamlit web app
├── train_model.py # Script to train and save model
├── spam_detector_lr.pkl # Trained logistic regression model
├── vectorizer.pkl # Saved TF-IDF vectorizer
├── requirements.txt # List of Python packages
└── README.md # Project documentation


---

## 📊 Dataset

- **Name:** SpamAssassin Public Corpus
- **Fields Used:**
  - `data`: Email content (text)
  - `label`: 0 (Not Spam), 1 (Spam)

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Yokeshvaran-S/Email_Spam_Detection.git
cd Email_Spam_Detection
