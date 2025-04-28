import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()sss

# API keys
OPENWEATHER_API_KEY = st.secrets["d800146b93a2ecf2ba158377ed11d44a"]
GROQ_API_KEY = st.secrets["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"]

# Inject CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load models and encoders
def load_files():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    with open('feature_encoders.pkl', 'rb') as f:
        feature_encoders = pickle.load(f)
    with open('fertilizer_recommendation_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, label_encoder, feature_encoders, model

# Fetch weather
def fetch_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        return temperature, humidity
    else:
        st.error(f"Error fetching weather: {data.get('message', 'Unknown error')}")
        return None, None

# Generate remarks using Groq
def generate_remark(fertilizer_name):
    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"Give a detailed agricultural advice for the fertilizer: {fertilizer_name}. Focus on why, when, and how to use it for farmers."
    chat_completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat_completion.choices[0].message.content

# Main App
def main():
    local_css("style.css")

    st.title("🚜 InfusionAI 3.0 — Smart Fertilizer Advisor")
    st.write("Empowering farmers with AI-driven fertilizer recommendations. 🌾")

    scaler, label_encoder, feature_encoders, model = load_files()

    city = st.text_input('Enter your City Name')

    if city:
        temperature, humidity = fetch_weather(city)
        if temperature is not None:
            st.success(f"Weather fetched successfully: {temperature}°C, {humidity}% humidity")

            moisture = st.number_input('Moisture (%)', min_value=0.0, max_value=100.0, value=30.0)
            soil_type = st.selectbox('Soil Type', feature_encoders['Soil Type'].classes_)
            crop_type = st.selectbox('Crop Type', feature_encoders['Crop Type'].classes_)

            nitrogen = st.number_input('Nitrogen Level (N)', min_value=0.0, max_value=100.0, value=20.0)
            phosphorus = st.number_input('Phosphorus Level (P)', min_value=0.0, max_value=100.0, value=30.0)
            potassium = st.number_input('Potassium Level (K)', min_value=0.0, max_value=100.0, value=40.0)

            if st.button('🌟 Recommend Fertilizer'):
                soil_encoded = feature_encoders['Soil Type'].transform([soil_type])[0]
                crop_encoded = feature_encoders['Crop Type'].transform([crop_type])[0]

                input_data = np.array([[
                    temperature, humidity, moisture,
                    soil_encoded, crop_encoded,
                    nitrogen, phosphorus, potassium
                ]])

                input_scaled = scaler.transform(input_data)
                prediction_encoded = model.predict(input_scaled)
                prediction = label_encoder.inverse_transform(prediction_encoded)[0]

                remark = generate_remark(prediction)

                st.success(f"🎯 Recommended Fertilizer: **{prediction}**")
                st.info(f"📝 Remark: {remark}")

if __name__ == '__main__':
    main()
