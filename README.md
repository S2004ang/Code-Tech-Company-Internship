# AI Chatbot Project

This is a complete AI Chatbot project implemented using Python, Natural Language Processing (NLP) with NLTK, and scikit-learn.

## Features
- Text Preprocessing using NLTK (Tokenization, Lemmatization)
- Greeting detection
- Information retrieval using TF-IDF Vectorization and Cosine Similarity
- Runs interactively in the terminal

## Project Structure
```
AI_Chatbot_Project/
│
├── chatbot.py           # Main chatbot logic
├── download_nltk.py     # Script to download necessary NLTK corpora
├── requirements.txt     # Required dependencies
│
├── data/
│   └── dataset.txt      # Text dataset the chatbot reads its knowledge from
│
└── README.md            # Project documentation
```

## Setup Instructions for VS Code

1. **Open the project in VS Code**
   Open VS Code, go to `File > Open Folder`, and select the `AI_Chatbot_Project` folder.

2. **Create a Virtual Environment (Optional but recommended)**
   Open the internal terminal (`Ctrl + \`` or `Terminal > New Terminal`) and run:
   ```bash
   python -m venv venv
   ```
   Activate the virtual environment:
   - On Windows: `.\venv\Scripts\activate`
   - On Mac/Linux: `source venv/bin/activate`

3. **Install Dependencies**
   Run the following command to install required packages from the VS Code terminal:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**
   Run the setup script to download the necessary NLTK components for tokenization and lemmatization:
   ```bash
   python download_nltk.py
   ```

5. **Run the Chatbot**
   Start the chatbot by running:
   ```bash
   python chatbot.py
   ```

## Usage Example
```
Welcome to the AI Chatbot Project!
Loading knowledge base...

ROBO: My name is Robo. I will answer your queries based on my dataset.
ROBO: If you want to exit, type 'bye'.

You: hi
ROBO: hello
You: what is python?
ROBO: python is a programming language.
You: what does nlp mean?
ROBO: nlp means natural language processing.
You: bye
ROBO: Bye! take care..
```
