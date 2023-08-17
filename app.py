import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Load the trained model from a pickle file
with open('Random Forest.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Custom CSS styles
st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .prediction {
        text-align: center;
        font-size: 28px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title('Diabetes Prediction App')

    st.markdown('<div class="title">Enter the following information for diabetes prediction:</div>', unsafe_allow_html=True)

    # Input fields
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=80)
    bmi = st.number_input('BMI', min_value=0, max_value=60, value=25)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.0, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=120, value=30)

    # Display dtype of node array
    st.write("Model's Node Array Dtype:", model._tree.node.dtype)


    # Predict button (updated to scale input data)
    if st.button('Predict'):
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
        scaled_input = scaler.fit_transform(input_data)  # Scale input data
        prediction = model.predict(scaled_input)
        if prediction == 0:
            st.markdown('<div class="prediction">Prediction: No Diabetes</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction">Prediction: Diabetes</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()




