#1: IMPORT DATA AND PERFORM EXPLORATORY DATA ANALYSIS 

# Import Libraries.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data.
reviews_df = pd.read_csv('amazon_reviews.csv')
reviews_df

# View the DataFrame Information.
reviews_df.info()

# View DataFrame Statistical Summary.
reviews_df.describe()

# Plot the count plot for the ratings.
sns.countplot(x = reviews_df['rating']) 

# Let's get the length of the verified_reviews column.
reviews_df['length'] = reviews_df['verified_reviews'].apply(len)

reviews_df

# Plot the histogram for the length.
reviews_df['length'].plot(bins=100, kind='hist') 

# Apply the describe method to get statistical summary.
reviews_df.describe()

# Plot the countplot for feedback.
# Positive ~2800
# Negative ~250
sns.countplot(x = reviews_df['feedback'])

#2: PLOT WORDCLOUD.

# Obtain only the positive reviews.
positive = reviews_df[reviews_df['feedback'] == 1]
positive

# Obtain the negative reviews only.
negative = reviews_df[reviews_df['feedback'] == 0]
negative

# Convert to list format.
sentences = positive['verified_reviews'].tolist()
len(sentences)

# Join all reviews into one large string.
sentences_as_one_string =" ".join(sentences)
sentences_as_one_string

from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))

sentences = negative['verified_reviews'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)
plt.figure(figsize = (20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))

#3: PERFORM DATA CLEANING.

# Let's define a pipeline to clean up all the messages.
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords.

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean

# Let's test the newly added function.
reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)

# show the original review.
print(reviews_df['verified_reviews'][5]) 

# show the cleaned up version.
print(reviews_df_clean[5])

from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier.
vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])

print(vectorizer.get_feature_names())

print(reviews_countvectorizer.toarray())  

reviews_countvectorizer.shape

reviews = pd.DataFrame(reviews_countvectorizer.toarray())

X = reviews

y = reviews_df['feedback']
y

#4: TRAIN AND TEST AI/ML MODELS.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

# Predicting the Test set results.
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))

