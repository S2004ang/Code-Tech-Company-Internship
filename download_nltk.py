import nltk

def download_packages():
    """Download necessary NLTK datasets for the chatbot"""
    print("Downloading NLTK packages...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    print("Download complete.")

if __name__ == "__main__":
    download_packages()
