# Email-spam-dection
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('/kaggle/input/spam-email/spam.csv')

# Data Preprocessing
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

# Create a pipeline for vectorization and model
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train the model
clf.fit(X_train, y_train)

# Predict on sample emails
emails = [
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]
predictions = clf.predict(emails)

# Model performance evaluation
accuracy = clf.score(X_test, y_test)

# Output predictions and accuracy
print("Predictions: ", predictions)  # 0 for ham, 1 for spam
print("Model Accuracy: ", accuracy)
