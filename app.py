from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

#importing pickle files
model = pickle.load(open('model.pkl','rb'))
scalar = pickle.load(open('scalar.pkl','rb'))
owner1 = pickle.load(open('owner.pkl','rb'))
intent1 = pickle.load(open('intent.pkl','rb'))
grade1 = pickle.load(open('grade.pkl','rb'))

def process(age, income, owner, exp, intent, grade, amt, rate):
    # encoding
    owner = owner1.transform([owner])
    intent = intent1.transform([intent])
    grade = grade1.transform([grade])

    # min max scalar
    input = np.array([age, income, int(owner), exp, int(intent), int(grade), amt, rate])

    scaled = scalar.transform([input])

    # predicting
    df = pd.DataFrame(scaled, columns=['person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
                                       'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate'])
    result = model.predict(df)
    return result

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        age = request.form["age"]
        income = request.form["income"]
        owner = request.form["Home_Ownership"]
        exp = request.form["Employment_length"]
        intent = request.form["loan_intent"]
        grade = request.form["loan_grade"]
        amt = request.form["loan_amount"]
        rate = request.form["loan_int_rate"]
        result = process(age, income, owner, exp, intent, grade, amt, rate)

        return render_template('predict.html',pred = result)

if __name__ == "__main__":
    app.run(debug=True)