import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

flask_app = Flask(__name__)
CLFmodel = pickle.load(open('lung_cancer_model.pkl', 'rb'))  # load the ML model

# The route() decorator to tell Flask what URL should trigger our function.
# ‘/’ is the root of the website, such as www.westga.edu
@flask_app.route("/")   
def index():
    return render_template("index.html")

# Prediction route
@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data from request
        age = float(request.form['Age'])
        gender = float(request.form['Gender'])
        air_pollution = float(request.form['Air_Pollution'])
        alcohol_use = float(request.form['Alcohol_Use'])
        dust_allergy = float(request.form['Dust_Allergy'])
        occupational_hazards = float(request.form['Occupational_Hazards'])
        genetic_risk = float(request.form['Genetic_Risk'])
        chronic_lung_disease = float(request.form['Chronic_Lung_Disease'])
        balanced_diet = float(request.form['Balanced_Diet'])
        obesity = float(request.form['Obesity'])
        smoking = float(request.form['Smoking'])
        passive_smoker = float(request.form['Passive_Smoker'])
        chest_pain = float(request.form['Chest_Pain'])
        coughing_of_blood = float(request.form['Coughing_of_Blood'])
        fatigue = float(request.form['Fatigue'])
        weight_loss = float(request.form['Weight_Loss'])
        shortness_of_breath = float(request.form['Shortness_of_Breath'])
        wheezing = float(request.form['Wheezing'])
        swallowing_difficulty = float(request.form['Swallowing_Difficulty'])
        clubbing_of_finger_nails = float(request.form['Clubbing_of_Finger_Nails'])
        frequent_cold = float(request.form['Frequent_Cold'])
        dry_cough = float(request.form['Dry_Cough'])
        snoring = float(request.form['Snoring'])

        

        

        # Create a NumPy array with the input features
        features = np.array([age, gender, air_pollution, alcohol_use, dust_allergy,
                             occupational_hazards, genetic_risk, chronic_lung_disease,
                             balanced_diet, obesity, smoking, passive_smoker,
                             chest_pain, coughing_of_blood, fatigue, weight_loss,
                             shortness_of_breath, wheezing, swallowing_difficulty,
                             clubbing_of_finger_nails, frequent_cold, dry_cough, snoring]).reshape(1, -1)

        # Perform prediction using the loaded model
        result = CLFmodel.predict(features)

        # Map numerical prediction to corresponding class labels if needed
        class_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        result_label = class_mapping.get(result[0], str(result[0]))

        # Render the result in the HTML template
        return render_template("index.html", predicted_text=result_label)

    except Exception as e:
        # Handle exceptions or errors (e.g., invalid input)
        error_message = f"Error: {str(e)}"
        return render_template("index.html", predicted_text=error_message)


if __name__ =="__main__":
    flask_app.run(debug = True)
