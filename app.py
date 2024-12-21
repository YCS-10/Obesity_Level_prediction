import warnings
from transformers import pipeline
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 

# Ignore warnings related to transformers library
warnings.filterwarnings("ignore")

app = Flask(__name__)

def load_data():
    # Load the dataset
    df = pd.read_csv("C:\\Users\\GANESH01\\Downloads\\ObesityDataSet_raw_and_data_sinthetic.csv")
    
    # Replace categorical values with numbers manually (instead of using LabelEncoder)
    df["Gender"] = df["Gender"].replace({"Female": 0, "Male": 1})
    df["family_history_with_overweight"] = df["family_history_with_overweight"].replace({"no": 0, "yes": 1})
    df["FAVC"] = df["FAVC"].replace({"no": 0, "yes": 1})
    
    # Handle MTRANS: Combine low frequency categories into 'Other'
    df["MTRANS"] = df["MTRANS"].replace({
        "Public_Transportation": 0, 
        "Automobile": 1, 
        "Walking": 2, 
        "Motorbike": 3, 
        "Bike": 3  # Merge Motorbike and Bike into "Other"
    })
    
    # CAEC: Map the eating habits to numbers
    df["CAEC"] = df["CAEC"].replace({
        "Sometimes": 0, 
        "Frequently": 1, 
        "Always": 2, 
        "no": 3
    })
    
    # CALC: Map the calorie consumption to numbers
    df["CALC"] = df["CALC"].replace({
        "Sometimes": 0, 
        "no": 1, 
        "Frequently": 2, 
        "Always": 3
    })
    
    # SMOKE, SCC, etc.
    df["SMOKE"] = df["SMOKE"].replace({"no": 0, "yes": 1})
    df["SCC"] = df["SCC"].replace({"no": 0, "yes": 1})
    
    # Map the target variable "NObeyesdad" with updated categories
    df["NObeyesdad"] = df["NObeyesdad"].replace({
        "Normal_Weight": 0, 
        "Overweight_Level_I": 1, 
        "Overweight_Level_II": 2, 
        "Obesity_Type_I": 3, 
        "Obesity_Type_II": 4, 
        "Obesity_Type_III": 5,
        "Insufficient_Weight": 6  # Add the new category
    })

    # Ensure there are no NaN values in the target variable
    df = df.dropna(subset=["NObeyesdad"])

    # Ensure that all target labels are valid (no unexpected categories)
    valid_labels = {0, 1, 2, 3, 4, 5, 6}
    df = df[df["NObeyesdad"].isin(valid_labels)]

    # Features and target
    x = df.iloc[:, :-1]  # Features
    y = df["NObeyesdad"]  # Target (NObeyesdad)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_scaled, y)

    return model, scaler


# Load model and scaler
model, scaler = load_data()

# Initialize GPT-2 text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Route for homepage
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input data from the form
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        gender = int(request.form['gender'])  # 0 for Female, 1 for Male
        family_history = int(request.form['family_history'])  # 0 for no, 1 for yes
        favc = int(request.form['favc'])  # 0 for no, 1 for yes
        fcvc = int(request.form['fcvc'])  # Values for FCVC (1-3)
        ncp = int(request.form['ncp'])  # Values for NCP (1-3)
        caec = int(request.form['caec'])  # 0 for no, 1 for yes, 2 for Sometimes
        smoke = int(request.form['smoke'])  # 0 for no, 1 for yes
        ch2o = int(request.form['ch2o'])  # Values for CH2O (1-3)
        scc = int(request.form['scc'])  # 0 for no, 1 for yes
        faf = int(request.form['faf'])  # Values for FAF (0-3)
        tue = int(request.form['tue'])  # Values for TUE (0-3)
        calc = int(request.form['calc'])  # 0 for no, 1 for yes
        mtrans = int(request.form['mtrans'])  # 0 for Public_Transportation, 1 for Walking

        # Prepare the input for prediction
        input_data = [
            [age, height, weight, gender, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]
        ]
        scaled_input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input_data)[0]

        # Map predictions back to categories
        remedies = {
            0: "Insufficient_Weight",
            1: "Normal_Weight",
            2: "Obesity_Type_I",
            3: "Obesity_Type_II",
            4: "Obesity_Type_III",
            5: "Overweight_Level_I",
            6: "Overweight_Level_II"
        }
        remedy = remedies[int(prediction)]

        # remedy_prompt = f"""
        #                 I need 3 suggestions for the {remedy} i'm facing.
        #                 """
        remedy_prompt = f"""
"Can you provide 2-3 practical and actionable remedies to help balance or improve the condition of someone with {remedy}? Include suggestions related to diet, exercise, and lifestyle changes that can be easily followed."
                        """
        generator = pipeline("text-generation", model="gpt2")
        response = generator(remedy_prompt, max_length=300, num_return_sequences=1)
        generated_text = response[0]['generated_text']
        suggestions_ = generated_text.split("\n\n")[1:]  # Assuming suggestions start from index 1

        return render_template('index.html', prediction=remedy, suggestions=suggestions_)

    return render_template('index.html')

if __name__ == "__main__":  
    app.run(debug=True)
