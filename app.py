import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model
with open('logistic.pkl', 'rb') as f:
  model = pickle.load(f)

# HTML & CSS Template embedded directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Eligibility Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #cbd5e1;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            width: 100%;
            max-width: 700px;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
        }
        h1 {
            font-size: 26px;
            font-weight: 700;
            text-align: center;
            color: var(--text-main);
            margin-bottom: 8px;
        }
        .subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
            margin-bottom: 30px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-main);
        }
        input, select {
            padding: 10px 14px;
            border: 1px solid var(--border);
            border-radius: 8px;
            font-size: 14px;
            outline: none;
            transition: all 0.2s ease;
            background-color: #fff;
        }
        input:focus, select:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }
        button {
            margin-top: 25px;
            width: 100%;
            padding: 12px;
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-box {
            margin-top: 25px;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            font-size: 16px;
            font-weight: 600;
        }
        .approved {
            background-color: #d1fae5;
            color: #065f46;
            border: 1px solid #a7f3d0;
        }
        .rejected {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #fecaca;
        }
        @media (max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            .form-group.full-width {
                grid-column: span 1;
            }
            .container {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Loan Eligibility Predictor</h1>
        <p class="subtitle">Enter your details below to check loan approval status</p>
        
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="gender">Gender</label>
                    <select name="gender" id="gender" required>
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="married">Married</label>
                    <select name="married" id="married" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="dependents">Dependents</label>
                    <select name="dependents" id="dependents" required>
                        <option value="0">0</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3+</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="education">Education</label>
                    <select name="education" id="education" required>
                        <option value="1">Graduate</option>
                        <option value="0">Not Graduate</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="self_employed">Self Employed</label>
                    <select name="self_employed" id="self_employed" required>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="property_area">Property Area</label>
                    <select name="property_area" id="property_area" required>
                        <option value="2">Urban</option>
                        <option value="1">Semiurban</option>
                        <option value="0">Rural</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="applicant_income">Applicant Income ($)</label>
                    <input type="number" name="applicant_income" id="applicant_income" placeholder="e.g. 5000" required>
                </div>

                <div class="form-group">
                    <label for="coapplicant_income">Coapplicant Income ($)</label>
                    <input type="number" name="coapplicant_income" id="coapplicant_income" placeholder="e.g. 1500" required>
                </div>

                <div class="form-group">
                    <label for="loan_amount">Loan Amount ($)</label>
                    <input type="number" name="loan_amount" id="loan_amount" placeholder="e.g. 120" required>
                </div>

                <div class="form-group">
                    <label for="loan_amount_term">Loan Term (Months)</label>
                    <input type="number" name="loan_amount_term" id="loan_amount_term" value="360" required>
                </div>

                <div class="form-group full-width">
                    <label for="credit_history">Credit History</label>
                    <select name="credit_history" id="credit_history" required>
                        <option value="1">Good / Debts Paid (1.0)</option>
                        <option value="0">Bad / Unpaid Debts (0.0)</option>
                    </select>
                </div>
            </div>

            <button type="submit">Predict Eligibility</button>
        </form>

        {% if prediction is not none %}
            {% if prediction == 1 %}
                <div class="result-box approved">
                    🎉 Congratulations! Your Loan Application is likely to be <strong>APPROVED</strong>.
                </div>
            {% else %}
                <div class="result-box rejected">
                    ❌ Sorry, your Loan Application is likely to be <strong>REJECTED</strong>.
                </div>
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def home():
  prediction = None
  if request.method == 'POST':
    try:
      # Mapping input fields to the exact order expected by the model features
      gender = float(request.form['gender'])
      married = float(request.form['married'])
      dependents = float(request.form['dependents'])
      education = float(request.form['education'])
      self_employed = float(request.form['self_employed'])
      applicant_income = float(request.form['applicant_income'])
      coapplicant_income = float(request.form['coapplicant_income'])
      loan_amount = float(request.form['loan_amount'])
      loan_amount_term = float(request.form['loan_amount_term'])
      credit_history = float(request.form['credit_history'])
      property_area = float(request.form['property_area'])

      features = np.array([[
          gender,
          married,
          dependents,
          education,
          self_employed,
          applicant_income,
          coapplicant_income,
          loan_amount,
          loan_amount_term,
          credit_history,
          property_area,
      ]])

      prediction = int(model.predict(features)[0])
    except Exception as e:
      print(f'Error during prediction: {e}')

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
