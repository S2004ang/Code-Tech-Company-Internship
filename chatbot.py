import os
import random
import warnings
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress scikit-learn warnings for cleaner output
warnings.filterwarnings('ignore')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Greeting inputs and responses for simple matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "hey there", "hola")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def load_data(file_path):
    """Load the knowledge base from a text file and tokenize it into sentences."""
    if not os.path.exists(file_path):
        print(f"Error: Knowledge file '{file_path}' not found.")
        return []
    
    with open(file_path, 'r', encoding='utf8', errors='ignore') as f:
        raw_text = f.read().lower()
        
    # Tokenize the text into sentences
    # Example input: "Hello! I am an AI chatbot." -> Example output: ["hello! i am an ai chatbot."]
    sent_tokens = nltk.sent_tokenize(raw_text)
    return sent_tokens

def LemTokens(tokens):
    """Lemmatize the tokens to their base form."""
    return [lemmatizer.lemmatize(token) for token in tokens]

# Create a dictionary to remove punctuation
remove_punct_dict = dict((ord(punct), None) for punct in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

def LemNormalize(text):
    """Normalize text by removing punctuation, converting to lowercase, tokenizing and lemmatizing."""
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

def greeting(sentence):
    """Check if the user input is a greeting and return a random greeting response."""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

def generate_response(user_response, sent_tokens):
    """Generate a response using TF-IDF and Cosine Similarity."""
    chatbot_response = ''
    
    # Add user response to sentence tokens for vectorization
    sent_tokens.append(user_response)
    
    # Transform sentences to vectors
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    
    # Calculate cosine similarity between the user input and all other sentences
    vals = cosine_similarity(tfidf[-1], tfidf)
    
    # Get the index of the most similar sentence (excluding the user query itself at -1)
    idx = vals.argsort()[0][-2]
    
    # Sort similarities to get the score of the best match
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    
    if req_tfidf == 0:
        # If no similarity is found
        chatbot_response = chatbot_response + "Sorry, I don't understand."
    else:
        # If a match is found, use that sentence
        chatbot_response = chatbot_response + sent_tokens[idx]
        
    # Remove the user response from the tokens to keep knowledge base clean
    sent_tokens.pop(-1)
    
    return chatbot_response

def main():
    print("Welcome to the AI Chatbot Project!")
    print("Loading knowledge base...")
    
    # Load data from dataset.txt
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'dataset.txt')
    sent_tokens = load_data(data_path)
    
    if not sent_tokens:
        print("Exiting due to missing dataset file.")
        return
        
    print("\nROBO: My name is Robo. I will answer your queries based on my dataset.")
    print("ROBO: If you want to exit, type 'bye'.\n")
    
    # Chatbot loop
    while True:
        user_response = input("You: ").lower()
        
        if user_response != 'bye':
            if user_response in ['thanks', 'thank you']:
                print("ROBO: You are welcome..")
                break
            else:
                # Check for greeting first
                greeting_match = greeting(user_response)
                if greeting_match != None:
                    print("ROBO: " + greeting_match)
                else:
                    # Search dataset for response
                    print("ROBO: ", end="")
                    print(generate_response(user_response, sent_tokens))
        else:
            print("ROBO: Bye! take care..")
            break

if __name__ == "__main__":
    main()
