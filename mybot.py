import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (smaller for faster testing)
intents = [
    {"intent": "greeting", "patterns": ["Hi", "Hello", "Hey"], "response": "Hello! How can I assist you today?"},
    {"intent": "goodbye", "patterns": ["Bye", "Goodbye"], "response": "Goodbye! Have a great day."},
    {"intent": "reset_password", "patterns": ["I forgot my password", "Reset password"], "response": "To reset your password, visit our website and click on 'Forgot Password'."},
    {"intent": "support", "patterns": ["I need help", "Can you assist me?"], "response": "Sure! Can you please specify your issue?"},
]

# Preprocessing function
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetical characters
    return text.strip()

# Prepare the training data
print("Preparing training data...")
train_data = []
train_labels = []
for intent in intents:
    for pattern in intent['patterns']:
        train_data.append(preprocess(pattern))
        train_labels.append(intent['intent'])

# Display partial output: Training data prepared
print(f"Training data: {train_data[:3]}")  # Show first 3 training patterns
print(f"Training labels: {train_labels[:3]}")  # Show first 3 labels

# Use a simple CountVectorizer for faster execution
print("Vectorizing data...")
vectorizer = CountVectorizer(max_features=100)
X_train = vectorizer.fit_transform(train_data)

# Display partial output: Vocabulary
print(f"Vocabulary: {vectorizer.get_feature_names_out()}")

# Train a Logistic Regression model
print("Training the model...")
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train, train_labels)

# Display partial output: Training complete
print("Model training complete!")

# Function to predict the intent
def predict_intent(user_input):
    processed_input = preprocess(user_input)
    X_input = vectorizer.transform([processed_input])
    intent = classifier.predict(X_input)[0]
    return intent

# Chatbot function
def chatbot():
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        intent = predict_intent(user_input)
        response = next((i["response"] for i in intents if i["intent"] == intent), "I didn't understand that.")
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()
