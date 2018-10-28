import pickle

from flask import Flask, request, render_template
from flask.json import jsonify
from sklearn.externals import joblib

word_vectorizer = pickle.load(open('models/vectorizer.pickle', 'rb'))
model = joblib.load('models/model.sav')
app = Flask(__name__)


@app.route('/ui', methods=['GET', 'POST'])
def ui_check():
    comment = ''
    banned = False
    if 'comment' in request.form:
        comment = request.form['comment']
        X = word_vectorizer.transform([comment])
        prediction = model.predict(X)
        banned = int(prediction[0])
        banned = banned == 1
    return render_template('form.html', banned=banned, comment=comment)


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    data = [json_['comment']]
    X = word_vectorizer.transform(data)
    prediction = model.predict(X)
    return jsonify({'prediction': int(prediction[0])})


if __name__ == '__main__':
    app.run(port=8080, debug=True)
