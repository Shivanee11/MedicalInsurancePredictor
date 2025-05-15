import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# Custom CSS for styling the app and plots
st.markdown("""
    <style>
        .reportview-container {
            background-color: #f4f4f9;
        }
        .sidebar .sidebar-content {
            background-color: #e3e3e3;
        }
        .block-container {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #FFB6C1;  /* Light Pink color for the button */
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-weight: bold;
            border: none;
        }

        /* Styling the plot containers */
        .plot-container {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-top: 20px;
        }

        /* For the title of the plots */
        .stSubheader {
            font-size: 24px;
            font-weight: bold;
            color: #FF69B4;  /* Matching the button color */
            margin-bottom: 15px;
        }

        /* Specific styles for the distribution plot */
        .distribution-plot {
            background-color: #f0f4f7;
            border: 2px solid #FF69B4;  /* Light pink border */
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Age vs Charges Plot */
        .age-vs-charges-plot {
            background-color: #f0f4f7;
            border: 2px solid #FF69B4;  /* Light pink border */
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Correlation heatmap styling */
        .heatmap {
            background-color: #f0f4f7;
            border: 2px solid #FF69B4;  /* Light pink border */
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title('Medical Insurance Cost Predictor')
st.subheader('Predict your medical insurance costs based on age, BMI, smoking habits, and more.')
st.write("""
         This app predicts the medical insurance cost based on user input such as age, BMI, gender, and smoking habits.
         Please input the details below to get the estimated cost.
         """)

# Load the dataset
@st.cache_data
def load_data():
    if os.path.exists('insurance.csv'):
        data = pd.read_csv('insurance.csv')  # Ensure this file exists in your project folder
    else:
        # Provide a default dataset if the file doesn't exist
        st.warning('insurance.csv not found. Using a default dataset.')
        data = pd.DataFrame({
            'age': [19, 21, 22, 23, 25],
            'bmi': [27.9, 33.1, 28.5, 34.2, 29.1],
            'children': [0, 1, 2, 3, 1],
            'smoker': ['no', 'yes', 'no', 'yes', 'no'],
            'region': ['southeast', 'northwest', 'northeast', 'southwest', 'southeast'],
            'charges': [16884.9, 1725.55, 4449.46, 21984.47, 3846.48]
        })
    return data

data = load_data()

# Display an image (make sure the path is correct)
try:
    st.image('insurance_set_2.jpg', caption='Insurance Prediction Model', use_container_width=True)
except Exception as e:
    st.error(f"Error loading image: {e}")
    st.image('https://www.example.com/sample-image.png', caption='Insurance Prediction Model', use_container_width=True)

# Sidebar inputs
st.sidebar.header('Enter your details:')
age = st.sidebar.slider('Age', 18, 100, 30)
bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
children = st.sidebar.selectbox('Number of children', [0, 1, 2, 3, 4])
smoker = st.sidebar.selectbox('Smoker', ['Yes', 'No'])
region = st.sidebar.selectbox('Region', ['North', 'East', 'South', 'West'])  # Updated regions


# Preprocessing the dataset
def preprocess_data(data):
    label_encoder = LabelEncoder()
    data['smoker'] = label_encoder.fit_transform(data['smoker'])
    data['region'] = label_encoder.fit_transform(data['region'])
    return data

data = preprocess_data(data)

# Train a RandomForest model
X = data[['age', 'bmi', 'children', 'smoker', 'region']]  # Features
y = data['charges']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Global prediction variable
prediction = None

# Function for prediction
def make_prediction(age, bmi, children, smoker, region):
    smoker = 1 if smoker == 'Yes' else 0
    # Updated region encoding
    region_dict = {'North': 0, 'East': 1, 'South': 2, 'West': 3}
    region = region_dict.get(region, 0)  # Default to North if region is not found
    
    prediction = model.predict([[age, bmi, children, smoker, region]])
    return prediction[0]


# Placeholder for the prediction
if st.button('Predict'):
    prediction = make_prediction(age, bmi, children, smoker, region)
    st.success('Prediction complete!')
    st.write(f'Predicted Insurance Cost: ${prediction:.2f}')

    # Add user's input as a new row to the data for plotting
    smoker_encoded = 1 if smoker == 'Yes' else 0
    region_dict = {'Northeast': 0, 'Southeast': 1, 'Southwest': 2, 'Northwest': 3}
    region_encoded = region_dict.get(region, 0)

    user_row = pd.DataFrame([{
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker': smoker_encoded,
        'region': region_encoded,
        'charges': prediction
    }])

    updated_data = pd.concat([data, user_row], ignore_index=True)

    # Distribution of Insurance Charges (with your prediction)
    st.subheader('Distribution of Insurance Charges (with your prediction)')
    st.markdown('<div class="distribution-plot">', unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.histplot(updated_data['charges'], bins=30, kde=True, color='#ADD8E6')  # Light blue color for histogram
    plt.axvline(prediction, color='red', linestyle='--', linewidth=2, label='Your Prediction')
    plt.legend()
    plt.title('Distribution of Insurance Charges')
    st.pyplot(plt)
    st.markdown('</div>', unsafe_allow_html=True)


    # Age vs Charges Plot
    st.subheader('Age vs Charges (with your prediction)')
    st.markdown('<div class="age-vs-charges-plot">', unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=updated_data, palette='coolwarm', s=100)  # Customize the color palette
    plt.scatter(age, prediction, color='#FF69B4', s=100, label='Your Prediction', edgecolor='black')  # Custom color for prediction point
    plt.legend()
    plt.title('Age vs Charges')
    st.pyplot(plt)
    st.markdown('</div>', unsafe_allow_html=True)

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    st.markdown('<div class="heatmap">', unsafe_allow_html=True)
    numeric_data = updated_data.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', linewidths=1)  # Custom color map for heatmap
    st.pyplot(plt)
    st.markdown('</div>', unsafe_allow_html=True)
