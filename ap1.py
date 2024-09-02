import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return predicted_class

# Streamlit UI
st.title('Sentiment_Check')
st.write("This is a sentiment analysis App")

user_input = st.text_area("Enter text here:")
if st.button('Analyze Sentiment'):
    if user_input:
        sentiment_class = predict_sentiment(user_input)
        sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        st.write(f"Sentiment: {sentiment_labels[sentiment_class]}")
    else:
        st.write("Please enter some text to analyze.")
