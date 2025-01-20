import streamlit as st
import joblib


# Load the trained model for heating and cooling
try:
    model_heating = joblib.load('/Users/sahilgulati/PycharmProjects/EnergyEfficiency/main/heating_model.pkl')
    model_cooling = joblib.load('/Users/sahilgulati/PycharmProjects/EnergyEfficiency/main/cooling_model.pkl')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Define prediction function


def predict_loads(input_features):
    heating_pred = model_heating.predict([input_features])
    cooling_pred = model_cooling.predict([input_features])
    return heating_pred[0], cooling_pred[0]


st.title("Energy Efficiency Predictor")
st.write("Input building parameters to predict heating and cooling loads.")

# Input sliders for features
relative_compactness = st.slider("Relative Compactness", 0.5, 1.0, 0.8)
surface_area = st.slider("Surface Area", 500, 1000, 750)
wall_area = st.slider("Wall Area", 200, 500, 300)
roof_area = st.slider("Roof Area", 100, 500, 200)
overall_height = st.slider("Overall Height", 3.0, 7.0, 5.5)
orientation = st.selectbox("Orientation", [2, 3, 4, 5])
glazing_area = st.slider("Glazing Area", 0.0, 0.4, 0.2)
glazing_area_dist = st.selectbox("Glazing Area Distribution", [0, 1, 2, 3, 4])

# Predict Button
if st.button("Predict"):
    input_features = [
        relative_compactness, surface_area, wall_area, roof_area,
        overall_height, orientation, glazing_area, glazing_area_dist
    ]
    heating, cooling = predict_loads(input_features)
    st.write(f"Predicted Heating Load: {heating:.2f}")
    st.write(f"Predicted Cooling Load: {cooling:.2f}")
