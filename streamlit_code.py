#!/usr/bin/env python
# flight_app.py
# Streamlit Flight Price Prediction App (Using Pretrained Model)

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# LOAD TRAIN + TEST CSV (only for UI dropdown options)
# ---------------------------------------------------------
@st.cache_data
def load_data(train_path="train.csv", test_path="test.csv"):
    try:
        df_train = pd.read_csv(train_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"train.csv not found at {train_path}. Place your file there.")

    try:
        df_test = pd.read_csv(test_path)
    except:
        df_test = pd.DataFrame()

    return df_train, df_test


# ---------------------------------------------------------
# LOAD PRETRAINED ARTIFACTS
# ---------------------------------------------------------
def load_artifacts():
    model = joblib.load("flight_model.pkl")
    encoders = joblib.load("encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, encoders, scaler, features


# ---------------------------------------------------------
# PREDICT SINGLE SAMPLE
# ---------------------------------------------------------
def predict_sample(model, encoders, scaler, features,
                   airline, airline_code, source, destination,
                   departure, arrival, flight_class, stops,
                   duration, days_left):

    sample = {
        "airline": str(airline),
        "airline_code": str(airline_code),
        "source": str(source),
        "destination": str(destination),
        "departure": str(departure),
        "arrival": str(arrival),
        "class": str(flight_class),
        "stops": str(stops),
        "duration": float(duration),
        "days_left": float(days_left),
    }

    row = []
    for col in features:
        if col in ["duration", "days_left"]:
            row.append(sample[col])
        else:
            le = encoders[col]
            v = sample[col]
            row.append(le.transform([v])[0] if v in le.classes_ else -1)

    df_row = pd.DataFrame([row], columns=features)

    df_row[["duration", "days_left"]] = scaler.transform(
        df_row[["duration", "days_left"]].astype(float)
    )

    pred = model.predict(df_row)[0]
    return float(max(pred, 0))


# ---------------------------------------------------------
# MAIN STREAMLIT UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Flight Price Prediction", layout="centered")
    st.markdown("<div style='background-color:teal;padding:10px;border-radius:8px'><h2 style='color:black;text-align:center'>Flight Price Prediction</h2></div>", unsafe_allow_html=True)

    st.write("This app loads a pretrained Random Forest model and predicts flight prices using your input.")

    # Load CSV (used only to populate dropdown options)
    try:
        df_raw, df_test_raw = load_data()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Load trained model + preprocessing artifacts
    try:
        model, encoders, scaler, features = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load pretrained model files: {e}")
        return

    st.success("Pretrained model loaded successfully!")
    st.write("---")

    # Helper for dropdown fallback options
    def unique_vals(col, default_list):
        if col in df_raw.columns:
            vals = sorted(df_raw[col].astype(str).unique().tolist())
            return vals if len(vals) > 0 else default_list
        return default_list

    # Dropdown lists
    airline_options = unique_vals('airline', ['IndiGo','Air India','SpiceJet'])
    source_options = unique_vals('source', ['DEL','BOM','BLR'])
    dest_options = unique_vals('destination', ['BOM','DEL','BLR'])
    dep_options = unique_vals('departure', ['09:00','15:00'])
    arr_options = unique_vals('arrival', ['11:00','18:00'])
    class_options = unique_vals('class', ['Economy','Business'])
    stops_options = unique_vals('stops', ['0','1','2'])

    st.subheader("Full Filter / Input Panel")

    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox("Airline", airline_options, key="main_airline")
        airline_code = st.text_input("Airline code", "6E", key="main_code")
        source = st.selectbox("Source", source_options, key="main_source")
        destination = st.selectbox("Destination", dest_options, key="main_destination")
        _class = st.selectbox("Class", class_options, key="main_class")

    with col2:
        departure = st.selectbox("Departure time", dep_options, key="main_departure")
        arrival = st.selectbox("Arrival time", arr_options, key="main_arrival")
        stops = st.selectbox("Stops", stops_options, key="main_stops")
        duration = st.number_input("Duration (minutes)", min_value=0.0, value=60.0, key="main_duration")
        days_left = st.number_input("Days left to departure", min_value=0.0, value=10.0, key="main_daysleft")

    if st.button("Predict Price", key="predict_main"):
        try:
            predicted = predict_sample(
                model, encoders, scaler, features,
                airline, airline_code, source, destination,
                departure, arrival, _class, stops,
                duration, days_left
            )
            st.metric("Predicted price (approx.)", f"₹{predicted:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.write("---")
    st.write("Using pretrained model. No training is performed.")


# ---------------------------------------------------------
# START APP
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
