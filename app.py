from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import joblib

app = Flask(__name__)

# ML code

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def predict():
    df = pd.read_csv('data/lyrics.csv')
    df_data = df[['lyrics', 'is_drake']]
    # Features and Labels
    df_x = df_data['lyrics']
    df_y = df_data['is_drake']

    corpus = df_x
    cv = CountVectorizer(min_df=2)
    X = cv.fit_transform(corpus)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size = 0.33, random_state = 42)
    
    #Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    
    #Save Model
    joblib.dump(clf, 'model.pkl')
    print("Model dumped!")
    
    #ytb_model = open('spam_model.pkl', 'rb')
    clf = joblib.load('model.pkl')
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vector = cv.transform(data).toarray()
        my_prediction = clf.predict(vector)
    return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug=True)