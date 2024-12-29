import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Define intents and their patterns
intents = [
    {"intent": "greeting", "patterns": ["Hi", "Hello", "Hey", "Good morning"], "response": "Hello! How can I assist you today?"},
    {"intent": "goodbye", "patterns": ["Bye", "Goodbye", "See you", "Take care"], "response": "Goodbye! Have a great day."},
    {"intent": "reset_password", "patterns": ["I forgot my password", "How do I reset my password?"], "response": "To reset your password, visit our website and click on 'Forgot Password'."},
    {"intent": "support", "patterns": ["I need help", "Can you assist me?", "Help me", "I have a problem"], "response": "Sure! Can you please specify your issue?"},
    {"intent": "feedback", "patterns": ["I have feedback", "How can I give feedback?", "Where to give feedback?"], "response": "You can provide your feedback directly here, and we'll use it to improve our services."},
]

# Preprocessing function
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetical characters
    return text.strip()

# Prepare training data
train_data = []
train_labels = []
for intent in intents:
    for pattern in intent['patterns']:
        train_data.append(preprocess(pattern))
        train_labels.append(intent['intent'])

# Vectorize the text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)

# Train a Logistic Regression model
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, train_labels)

# Function to predict the intent
def predict_intent(user_input):
    processed_input = preprocess(user_input)
    X_input = vectorizer.transform([processed_input])
    intent = classifier.predict(X_input)[0]
    return intent

# Enhanced response generator
def generate_response(intent, user_input):
    if intent == "feedback":
        return f"Thank you for your feedback: '{user_input}'. We'll use it to improve our services."
    else:
        return next((i["response"] for i in intents if i["intent"] == intent), "I didn't understand that.")

# Streamlit interface
st.title("ConvoCraft")
st.write("Type your message below and press Enter to chat with the bot. Type 'exit' to quit.")

user_input = st.text_input("You:", "")
if user_input:
    if user_input.lower() == "exit":
        st.write("Chatbot: Goodbye!")
    else:
        intent = predict_intent(user_input)
        response = generate_response(intent, user_input)
        st.write(f"Chatbot: {response}")
