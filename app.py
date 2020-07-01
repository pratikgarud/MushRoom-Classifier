from flask import Flask,render_template,request
import pickle
import numpy

app = Flask(__name__)
cv = pickle.load(open('cv-transform.pkl','rb'))
classifier = pickle.load(open('Spam_Model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        msg = [message]
        vect = cv.transform(msg).toarray()
        prediction = classifier.predict(vect)
        return render_template('home.html',msg=prediction)


if __name__ == '__main__':
    app.run(debug=True)


