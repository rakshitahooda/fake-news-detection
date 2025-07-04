# fake-news-detection
# Detecting fake news using Logistic Regression and TF-IDF
# 📰 Fake News Detection using Machine Learning

This project uses a Logistic Regression model combined with TF-IDF vectorization to detect whether a news article is fake or real based on its title and text.

---

## 📌 Dataset
- Used: [Kaggle Fake & Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Also available through: [GitHub link](https://raw.githubusercontent.com/krishnaik06/Fake-News-Detection/master/news.csv)

---

## 🔧 Tech Stack
- Python
- Pandas, Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

## 📈 Model Performance
- **Accuracy**: ~94%
- **Evaluation**: Classification report with precision, recall, F1-score

---

## 📁 How to Run
1. Clone or download the repo
2. Open the notebook `fake_news_detection.ipynb` in Google Colab or Jupyter Notebook
3. Run all cells to train and evaluate the model

---

## ✅ Output Example

```python
Accuracy Score: 0.94
Classification Report:
               precision    recall  f1-score   support
        FAKE       0.94      0.95      0.94       590
        REAL       0.95      0.93      0.94       585
