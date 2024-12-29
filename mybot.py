import random
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (intents) with fewer patterns for faster testing
intents = [
    {"intent": "greeting", "patterns": ["Hi", "Hello", "Hey"], "response": "Hello! How can I assist you today?"},
    {"intent": "goodbye", "patterns": ["Bye", "Goodbye"], "response": "Goodbye! Have a great day."},
    {"intent": "reset_password", "patterns": ["How do I reset my password?", "I forgot my password"], "response": "To reset your password, visit our website and click on 'Forgot Password'."},
    {"intent": "support", "patterns": ["I need help", "Can you assist me?"], "response": "Sure! Can you please specify your issue?"},
]

# Preprocessing function (without nltk)
def preprocess(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words (1 or 2 letters)
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetical characters
    return text.strip()

# Prepare the data for training
train_data = []
train_labels = []
for intent in intents:
    for pattern in intent['patterns']:
        train_data.append(preprocess(pattern))
        train_labels.append(intent['intent'])

# Train a TF-IDF Vectorizer with optimizations
vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 1))
X_train = vectorizer.fit_transform(train_data)

# Train a Logistic Regression Classifier with solver='liblinear'
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, train_labels)

# Function to get the intent from user input
def predict_intent(user_input):
    processed_input = preprocess(user_input)
    X_input = vectorizer.transform([processed_input])
    intent = classifier.predict(X_input)[0]
    return intent

# Store user preferences (personalization)
user_data = {}

# Function to personalize user experience
def personalize(user_id, preference=None):
    if user_id not in user_data:
        user_data[user_id] = {}
    if preference:
        user_data[user_id].update(preference)
    return user_data[user_id]

# Function to handle user feedback (feedback loop)
def handle_feedback(feedback, user_id):
    if feedback == "helpful":
        print(f"User {user_id} found the help useful.")
    elif feedback == "unhelpful":
        print(f"User {user_id} found the help unhelpful. Let's improve!")

# Chatbot function to respond to user input
def chatbot(user_id, user_input):
    # Check if user has preferences
    user_preferences = user_data.get(user_id, {})

    # Predict intent based on user input
    intent = predict_intent(user_input)
    
    # Response based on intent
    response = ""
    for intent_data in intents:
        if intent_data["intent"] == intent:
            response = intent_data["response"]
            break
    
    # Provide personalized response if necessary
    if user_preferences.get('greeting_time') == 'morning' and intent == "greeting":
        response = "Good morning! How can I assist you today?"
    
    # For feedback system
    if "helpful" in user_input or "unhelpful" in user_input:
        feedback = "helpful" if "helpful" in user_input else "unhelpful"
        handle_feedback(feedback, user_id)
    
    return response

# Example conversation
def start_conversation():
    user_id = random.randint(1000, 9999)  # Simulating a random user ID
    print(f"Chatbot initialized. User ID: {user_id}")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye! Have a great day.")
            break
        
        # Chatbot responds based on intent
        response = chatbot(user_id, user_input)
        print(f"Chatbot: {response}")
        
        # Ask for feedback
        feedback = input("Was this response helpful? (yes/no): ").lower()
        if feedback == "yes":
            personalize(user_id, {"greeting_time": "morning"})
            print("Chatbot: Thank you for your feedback!")
        elif feedback == "no":
            print("Chatbot: Sorry about that. I'll improve based on your feedback.")
            handle_feedback("unhelpful", user_id)

if __name__ == "__main__":
    start_conversation()
