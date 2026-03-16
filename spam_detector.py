import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

data['label'] = data['label'].map({'ham':0,'spam':1})

X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2
)

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_counts, y_train)

predictions = model.predict(X_test_counts)

print("Accuracy:", accuracy_score(y_test, predictions))

email = ["Congratulations! You won a free lottery ticket"]
email_vector = vectorizer.transform(email)

prediction = model.predict(email_vector)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam")

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x=data['label'])
plt.title("Spam vs Ham Distribution")
plt.show()