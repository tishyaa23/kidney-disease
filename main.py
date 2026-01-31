import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.set_page_config(layout="wide")



def predict_from_xgb(input_values):
    """
    Predicts the CKD status using the trained XGBoost model.

    Args:
        input_values: A list of feature values in the same order as the training data.

    Returns:
        True if the prediction is positive for Chronic Kidney Disease, False otherwise.
    """
    try:
        with open("model.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
    except FileNotFoundError:
        st.error("Error: model.pkl not found. Please ensure the model file exists.")
        return None

    # Define the column names as per the training data
    columns = [
       'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
       'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
       'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
       'potassium', 'haemoglobin', 'packed_cell_volume',
       'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
       'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
       'peda_edema', 'aanemia'
    ]

    # Convert input values to a DataFrame
    input_df = pd.DataFrame([input_values], columns=columns)

    # Make prediction
    prediction = loaded_model.predict(input_df)[0]
    
    return prediction   # Assuming 1 represents CKD

# Streamlit app layout
st.title("Chronic Kidney Disease Prediction")

st.sidebar.header("Input Features")

# Collecting user inputs through sidebar
age = st.sidebar.number_input("Age", min_value=0.0, max_value=120.0, value=27.0)
blood_pressure = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=60.0)
specific_gravity = st.sidebar.number_input("Specific Gravity", format="%.3f", min_value=1.0, max_value=1.2, value=1.015)
albumin = st.sidebar.number_input("Albumin (g/dL)", format="%.1f", min_value=0.0, max_value=10.0, value=0.0)
sugar = st.sidebar.number_input("Sugar (g/dL)", format="%.1f", min_value=0.0, max_value=10.0, value=0.0)
red_blood_cells = st.sidebar.selectbox("Red Blood Cells", options=['normal', 'abnormal'], index=1)  # 1 -> normal
pus_cell = st.sidebar.selectbox("Pus Cell", options=['normal', 'abnormal'], index=1)  # 1 -> normal
pus_cell_clumps = st.sidebar.selectbox("Pus Cell Clumps", options=['present', 'not_present'], index=1)  # 0 -> not_present
bacteria = st.sidebar.selectbox("Bacteria", options=['present', 'not_present'], index=1)  # 0 -> not_present
blood_glucose_random = st.sidebar.number_input("Blood Glucose Random (mg/dL)", format="%.1f", min_value=0.0, max_value=300.0, value=76.0)
blood_urea = st.sidebar.number_input("Blood Urea (mg/dL)", format="%.1f", min_value=0.0, max_value=300.0, value=44.0)
serum_creatinine = st.sidebar.number_input("Serum Creatinine (mg/dL)", format="%.1f", min_value=0.0, max_value=10.0, value=3.9)
sodium = st.sidebar.number_input("Sodium (mEq/L)", format="%.1f", min_value=0.0, max_value=200.0, value=127.0)
potassium = st.sidebar.number_input("Potassium (mEq/L)", format="%.1f", min_value=0.0, max_value=10.0, value=4.3)
haemoglobin = st.sidebar.number_input("Haemoglobin (g/dL)", format="%.1f", min_value=0.0, max_value=20.0, value=7.3)
packed_cell_volume = st.sidebar.number_input("Packed Cell Volume (%)", format="%.1f", min_value=0.0, max_value=100.0, value=26.0)
white_blood_cell_count = st.sidebar.number_input("White Blood Cell Count (cells/μL)", format="%.1f", min_value=0.0, max_value=20000.0, value=6300.0)
red_blood_cell_count = st.sidebar.number_input("Red Blood Cell Count (million cells/μL)", format="%.1f", min_value=0.0, max_value=10.0, value=4.4)
hypertension = st.sidebar.selectbox("Hypertension", options=['yes', 'no'], index=1)  # 0 -> no
diabetes_mellitus = st.sidebar.selectbox("Diabetes Mellitus", options=['yes', 'no'], index=1)  # 0 -> no
coronary_artery_disease = st.sidebar.selectbox("Coronary Artery Disease", options=['yes', 'no'], index=1)  # 0 -> no
appetite = st.sidebar.selectbox("Appetite", options=['good', 'poor'], index=1)  # 1 -> poor
peda_edema = st.sidebar.selectbox("Pedal Edema", options=['yes', 'no'], index=0)  # 1 -> yes
aanemia = st.sidebar.selectbox("Anemia", options=['yes', 'no'], index=0)  # 1 -> yes

# Prepare the input list for prediction
input_values = [
    age, blood_pressure, specific_gravity, albumin, sugar,
    1 if red_blood_cells == 'normal' else 0,
    1 if pus_cell == 'normal' else 0,
    1 if pus_cell_clumps == 'present' else 0,
    1 if bacteria == 'present' else 0,
    blood_glucose_random, blood_urea, serum_creatinine, sodium,
    potassium, haemoglobin, packed_cell_volume,
    white_blood_cell_count, red_blood_cell_count,
    1 if hypertension == 'yes' else 0,
    1 if diabetes_mellitus == 'yes' else 0,
    1 if coronary_artery_disease == 'yes' else 0,
    1 if appetite == 'poor' else 0,
    1 if peda_edema == 'yes' else 0,
    1 if aanemia == 'yes' else 0
]

# Button to make prediction
predict_button = st.sidebar.button("Predict")

# Create two columns on the homepage
col1, col2 = st.columns(2)

# Visualization of numerical input features (excluding binary features)
numerical_features = {
    'age': age,
    'blood_pressure': blood_pressure,
    'specific_gravity': specific_gravity,
    'albumin': albumin,
    'sugar': sugar,
    'blood_glucose_random': blood_glucose_random,
    'blood_urea': blood_urea,
    'serum_creatinine': serum_creatinine,
    'sodium': sodium,
    'potassium': potassium,
    'haemoglobin': haemoglobin,
    'packed_cell_volume': packed_cell_volume,
    'white_blood_cell_count': white_blood_cell_count,
    'red_blood_cell_count': red_blood_cell_count
}

# Define realistic min and max values for scaling
scaling_values = {
    'age': (0, 120),
    'blood_pressure': (0, 200),
    'specific_gravity': (1.0, 1.2),
    'albumin': (0.0, 10.0),
    'sugar': (0.0, 10.0),
    'blood_glucose_random': (0.0, 300.0),
    'blood_urea': (0.0, 300.0),
    'serum_creatinine': (0.0, 10.0),
    'sodium': (0.0, 200.0),
    'potassium': (0.0, 10.0),
    'haemoglobin': (0.0, 20.0),
    'packed_cell_volume': (0.0, 100.0),
    'white_blood_cell_count': (0.0, 20000.0),
    'red_blood_cell_count': (0.0, 10.0)
}

# Convert numerical features to DataFrame and scale them
numerical_df = pd.DataFrame(list(numerical_features.items()), columns=['Feature', 'Value'])

# Apply scaling based on predefined ranges
numerical_df['Scaled Value'] = numerical_df.apply(
    lambda row: (row['Value'] - scaling_values[row['Feature']][0]) / 
                (scaling_values[row['Feature']][1] - scaling_values[row['Feature']][0]), 
    axis=1
)

# Create radar chart
fig = px.line_polar(
    numerical_df,
    r='Scaled Value',
    theta='Feature',
    line_close=True,
    title="Numerical Input Features Visualization",
    template="plotly_dark"
)

# Show the radar chart in the first column
col1.plotly_chart(fig)

# Make prediction and show result in the second column
if predict_button:
    prediction = predict_from_xgb(input_values)
    if prediction is not None:
        if prediction==1:
            result_text = "The model predicts that you **have Chronic Kidney Disease (CKD)**."
            col2.error(result_text)
        else:
            result_text = "The model predicts that you **do not have Chronic Kidney Disease (CKD)**."
            col2.success(result_text)
            st.balloons()
        
