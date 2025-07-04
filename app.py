import streamlit as st
import pickle

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
        vec_input = vectorizer.transform([user_input])
        prediction = model.predict(vec_input)[0]

        if prediction == "FAKE":
            st.error("🚫 This news is FAKE.")
        else:
            st.success("✅ This news is REAL.")
