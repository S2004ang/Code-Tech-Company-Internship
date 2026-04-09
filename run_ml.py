import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style('whitegrid')

print("Fetching Dataset...")
url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
print("Dataset Shape:", df.shape)

print("\nMissing Values:\n", df.isnull().sum())

# Encode labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
print("\nDistribution of labels:")
print(df['label'].value_counts())

print("\nSplitting Dataset...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
print("Training Set Size:", X_train.shape[0])
print("Testing Set Size:", X_test.shape[0])

print("\nExtracting Features using TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for model_name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc
    
    print(f"\n{'='*40}")
    print(f"{model_name} Model Evaluation")
    print(f"{'='*40}")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

print("\nGenerating Model Accuracy Comparison Chart...")
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)
plt.savefig('accuracy_comparison.png')
print("-> accuracy_comparison.png saved!")

print("\nGenerating Confusion Matrix for Naive Bayes...")
nb_model = models["Naive Bayes"]
y_pred_nb = nb_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix - Naive Bayes')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix_nb.png')
print("-> confusion_matrix_nb.png saved!")

print("\nExecution Completed Successfully!")
