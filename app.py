import streamlit as st
import pickle
import requests

# Function to check NewsAPI
def check_with_newsapi(query):
    api_key = "21c259a09cca476894f5dba48115a09f"  # 🔁 Replace with your real API key
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&pageSize=5&apiKey={api_key}"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    return articles

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Fake News Detection App", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article below to check if it's REAL or FAKE.")

user_input = st.text_area("Paste news text here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        st.info("🔎 Checking real news sources...")
        articles = check_with_newsapi(user_input)

        if len(articles) > 0:
            st.success("✅ This news appears in real sources.")
            for article in articles[:2]:
                st.write(f"- [{article['title']}]({article['url']})")
        else:
            st.warning("⚠️ No matching news found in real sources. Using ML model...")
            vec_input = vectorizer.transform([user_input])
            prediction = model.predict(vec_input)[0]

            if prediction == "FAKE":
                st.error("🚫 This news is FAKE.")
            else:
                st.success("✅ This news is REAL.")

st.caption("⚠️ This result is based on text patterns only, not real-time fact-checking.")

