import random
import re

# Sample dataset (intents)
intents = [
    {"intent": "greeting", "patterns": ["hi", "hello", "hey", "good morning", "how are you?"], "response": "Hello! How can I assist you today?"},
    {"intent": "goodbye", "patterns": ["bye", "goodbye", "see you", "take care"], "response": "Goodbye! Have a great day."},
    {"intent": "reset_password", "patterns": ["how do I reset my password?", "I forgot my password", "help me with password reset"], "response": "To reset your password, visit our website and click on 'Forgot Password'."},
    {"intent": "support", "patterns": ["I need help", "can you assist me?", "help me", "I have a problem"], "response": "Sure! Can you please specify your issue?"},
]

# Preprocessing function (without nltk)
def preprocess(text):
    # Convert to lowercase and remove non-alphabetical characters
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetical characters
    return text.strip()

# Simple function to match the input to the closest intent
def match_intent(user_input):
    processed_input = preprocess(user_input)
    
    for intent in intents:
        for pattern in intent['patterns']:
            # Compare user input to each pattern (case insensitive)
            if re.search(r'\b' + re.escape(processed_input) + r'\b', preprocess(pattern)):
                return intent['response']
    
    return "Sorry, I didn't understand that. Can you please clarify?"

# Store user preferences (personalization)
user_data = {}

# Function to handle user feedback (feedback loop)
def handle_feedback(feedback, user_id):
    if feedback == "helpful":
        print(f"User {user_id} found the help useful.")
    elif feedback == "unhelpful":
        print(f"User {user_id} found the help unhelpful. Let's improve!")

# Chatbot function to respond to user input
def chatbot(user_id, user_input):
    # Predict intent and get response
    response = match_intent(user_input)
    
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
            print("Chatbot: Thank you for your feedback!")
        elif feedback == "no":
            print("Chatbot: Sorry about that. I'll improve based on your feedback.")

if __name__ == "__main__":
    start_conversation()
