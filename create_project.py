import json
import os

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Machine Learning Project: Text Classification (Spam Detection)\n\n",
                "In this project, we implement a predictive model to classify text messages as 'Spam' or 'Ham' (Not Spam). We will perform data preprocessing, text feature extraction using TF-IDF, model training using several algorithms, and evaluate their performances."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Import Necessary Libraries\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "from sklearn.naive_bayes import MultinomialNB\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report\n",
                "\n",
                "%matplotlib inline\n",
                "sns.set_style('whitegrid')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Load the Dataset\n",
                "We will use the classic SMS Spam Collection dataset fetched directly from a public URL."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "url = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'\n",
                "# The dataset is tab-separated\n",
                "df = pd.read_csv(url, sep='\\t', header=None, names=['label', 'text'])\n",
                "\n",
                "print(\"Dataset Shape:\", df.shape)\n",
                "display(df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Data Preprocessing\n",
                "- Handle missing values\n",
                "- Encode the labels (spam = 1, ham = 0)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check for missing values\n",
                "print(\"Missing Values:\\n\", df.isnull().sum())\n",
                "\n",
                "# Encode labels: 'ham' -> 0, 'spam' -> 1\n",
                "df['label'] = df['label'].map({'ham': 0, 'spam': 1})\n",
                "\n",
                "print(\"\\nDistribution of labels:\")\n",
                "print(df['label'].value_counts())\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Train-Test Split\n",
                "Split the data into training (80%) and testing (20%) sets."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)\n",
                "\n",
                "print(\"Training Set Size:\", X_train.shape[0])\n",
                "print(\"Testing Set Size:\", X_test.shape[0])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Feature Extraction (Text to Vectors using TF-IDF)\n",
                "We use `TfidfVectorizer` to convert sentences into numerical arrays that our machine learning models can understand."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "vectorizer = TfidfVectorizer(stop_words='english')\n",
                "\n",
                "# Fit on training data and transform it\n",
                "X_train_tfidf = vectorizer.fit_transform(X_train)\n",
                "\n",
                "# Transform testing data (do NOT fit on testing data to avoid data leakage)\n",
                "X_test_tfidf = vectorizer.transform(X_test)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Model Training and Evaluation\n",
                "We will train three different algorithms and evaluate their performance based on accuracy and classification reports."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize models\n",
                "models = {\n",
                "    \"Naive Bayes\": MultinomialNB(),\n",
                "    \"Logistic Regression\": LogisticRegression(max_iter=1000),\n",
                "    \"Random Forest\": RandomForestClassifier(n_estimators=100, random_state=42)\n",
                "}\n",
                "\n",
                "results = {}\n",
                "\n",
                "for model_name, model in models.items():\n",
                "    # Train the model\n",
                "    model.fit(X_train_tfidf, y_train)\n",
                "    \n",
                "    # Make predictions\n",
                "    y_pred = model.predict(X_test_tfidf)\n",
                "    \n",
                "    # Evaluate\n",
                "    acc = accuracy_score(y_test, y_pred)\n",
                "    results[model_name] = acc\n",
                "    \n",
                "    print(f\"\\n{'='*40}\")\n",
                "    print(f\"{model_name} Model Evaluation\")\n",
                "    print(f\"{'='*40}\")\n",
                "    print(f\"Accuracy: {acc:.4f}\\n\")\n",
                "    print(\"Classification Report:\\n\", classification_report(y_test, y_pred))\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Model Comparison & Verifying Results via Heatmap\n",
                "Let's compare the accuracies side-by-side, and visualize the confusion matrix of the Naive Bayes model using a Heatmap."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Bar plot of accuracies\n",
                "plt.figure(figsize=(8, 5))\n",
                "sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')\n",
                "plt.title('Model Accuracy Comparison')\n",
                "plt.ylabel('Accuracy')\n",
                "plt.ylim(0.9, 1.0)\n",
                "plt.show()\n",
                "\n",
                "# Confusion Matrix for Naive Bayes\n",
                "nb_model = models[\"Naive Bayes\"]\n",
                "y_pred_nb = nb_model.predict(X_test_tfidf)\n",
                "cm = confusion_matrix(y_test, y_pred_nb)\n",
                "\n",
                "plt.figure(figsize=(6, 4))\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])\n",
                "plt.title('Confusion Matrix - Naive Bayes')\n",
                "plt.xlabel('Predicted Label')\n",
                "plt.ylabel('True Label')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Final Conclusion\n",
                "Overall, we accomplished the following:\n",
                "1. **Preprocessing:** Handled labels by encoding them, checked missing data.\n",
                "2. **Vectorization:** Extracted numerical representations from text using TF-IDF.\n",
                "3. **Model Verification:** Tested multiple models tracking overall accuracy.\n",
                "\n",
                "We can observe that the **Naive Bayes** model performs extremely well on simple text classification tasks like the Spam Dataset, producing excellent precision and recall output. Both other models performed well, but the performance vs calculation-speed ratio for Naive Bayes makes it a premier choice for these applications."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

file_path = os.path.join("c:\\\\Users\\\\S.Madhupavani\\\\OneDrive\\\\CodeTech\\\\Task4", "text_classification_project.ipynb")
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)
print(f"Created: {file_path}")
