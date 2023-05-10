from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

if not os.path.exists('templates'):
    os.mkdir('templates')

# Declare a Flask app
app = Flask(__name__)

# Main function here
# ------------------

@app.route('/', methods=['GET', 'POST'])

def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        clf = joblib.load("model.pkl")
        
        # Get values through input bars
        age = float(request.form.get("age"))
        sex = float(request.form.get("sex"))
        bmi = float(request.form.get("bmi"))
        children = float(request.form.get("children"))
        smoker = float(request.form.get("smoker"))
        region = float(request.form.get("region"))
        
        # Put inputs to dataframe
        X = pd.DataFrame([[age, sex, bmi, children, smoker, region]], columns = ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        
        # Get prediction
        prediction = clf.predict(X)[0]
        
    else:
        prediction = ""
        
    return render_template("website.html", output = prediction)


# Running the app
if __name__ == '__main__':
    app.run(debug = True)
#app.run(port=5000)