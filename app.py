from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("MODEL/model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("welcome.html")

@app.route('/dashboard')
def dashboard():
    return render_template("dashboard.html")

@app.route('/form')
def form():
    return render_template("form.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    data = [
    float(request.form['Gender']),
    float(request.form['Married']),
    float(request.form['Dependents']),
    float(request.form['Education']),
    float(request.form['Self_Employed']),
    float(request.form['ApplicantIncome']),
    float(request.form['CoapplicantIncome']),
    float(request.form['LoanAmount']),
    float(request.form['Loan_Amount_Term']),
    float(request.form['Credit_History']),
    float(request.form['Property_Area'])
]
    
    # Convert to array
    final_input = np.array([data])
    
    # Predict
    prediction = model.predict(final_input)
    
    result = "Approved ✅" if prediction[0] == 1 else "Rejected ❌"
    
    return render_template("result.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
    