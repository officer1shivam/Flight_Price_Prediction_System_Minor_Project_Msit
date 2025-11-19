## 🧠 Flight Fare Price Prediction System

After collecting and stitching the scraped flight dataset, a dedicated Jupyter/Colab notebook was created named:

**`Flight Ticket Price Prediction.ipynb which later divided into 2 python files for ease : flight_app.py & streamlit_code.py`**

This notebook handles the **entire machine learning workflow**, from preprocessing to model evaluation and export.

---

## 🔧 1. Data Preprocessing Pipeline

The notebook performs all required preprocessing steps to convert raw scraped flight data into model-ready input.

### ✔ Missing Value Handling
- Categorical missing values are replaced with `"Unknown"`  
- Numerical missing values are imputed using the **median**  

### ✔ Removing Duplicates & Outliers
- Duplicate rows (excluding ID & target) are removed  
- IQR-based clipping is applied to numerical features (`duration`, `days_left`) to reduce extreme outliers  

### ✔ Feature Engineering
- **Airline code extraction** from the `flight` string (e.g., FR, W6)  
- Dropping unnecessary columns like `id` and `flight`

### ✔ Label Encoding
Each categorical column is encoded using **LabelEncoder**, fitted only on training data to avoid leakage.  
Encoders are stored later as `.pkl` files.

### ✔ Standardization
Numerical columns are scaled using `StandardScaler`:
- `duration`
- `days_left`

This ensures all models receive normalized inputs.

---

## 🤖 2. Model Training & Evaluation

Multiple regression models were trained:

- Linear Regression  
- Ridge & Lasso  
- Decision Tree  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- LightGBM  
- K-Nearest Neighbor  
- Polynomial Regression (degree 3)

For each model:
1. Fit on training data  
2. Predict on validation data  
3. Compute **RMSE** & **R²** metrics  
4. Store performance  
5. Identify best model  

### ✔ Hyperparameter Tuning  
GridSearchCV tuning was applied to:
- Random Forest  
- XGBoost  
- LightGBM  

The best-performing tuned model is automatically selected.

---

## 🏆 3. Selecting the Best Model

All model scores are collected in a comparison table.  
The model with the **lowest RMSE** is chosen as the final model.

The final model is then **retrained** using the full dataset  
(training + validation combined) for maximum accuracy.

---

## 💾 4. Exporting Model Files (to avoid retraining every time)

To make the model usable in UI apps or deployment, four essential components are saved:

| File Name        | Description |
|------------------|-------------|
| `flight_model.pkl` | The final trained ML model (best-performing one) |
| `scaler.pkl`        | StandardScaler used for normalizing numerical features |
| `encoders.pkl`      | Dictionary of LabelEncoders used for each categorical column |
| `features.pkl`      | Exact list & order of features expected by the model |

These files ensure:

### ✔ No need to retrain the model again  
Your Streamlit app or any backend service can load these `.pkl` files and instantly perform predictions.

### ✔ Guaranteed consistency  
The model, encoders, and scaler expect the data in the exact same format used during training.

This completes the **Data Processing + Machine Learning** part of the project.

## 🖥️ 5. Streamlit Deployment with Model Inference

After exporting all four `.pkl` files, a **Streamlit-based frontend** was developed to provide a clean, interactive interface where users can enter flight-related details and instantly get a predicted ticket price.

The Streamlit application loads:

- `flight_model.pkl` → Final trained regression model  
- `scaler.pkl` → For normalizing numerical inputs  
- `encoders.pkl` → To transform categorical user selections  
- `features.pkl` → To maintain correct feature ordering during prediction  

---

## 🎛️ 6. Streamlit User Interface (UI) Design

The UI was designed to be simple, intuitive, and beginner-friendly:

### ✔ Numerical Inputs (Manual Entry)
Users can directly input numerical fields such as:
- **Duration** (minutes)  
- **Days Left** before the flight  

These fields use `st.number_input()` for precise and error-free input.

### ✔ Categorical Inputs (Dropdown Select Options)
For categorical features, the app provides dropdown menus using `st.selectbox()`, ensuring users only select valid categories.  
Examples include:
- Airline  
- Departure Airport  
- Arrival Airport  
- Flight Class  
- Stops (Non-stop / 1-stop / Multi-stop)

This prevents invalid or unseen categories from being entered, ensuring the backend model receives acceptable values.

---

## 📤 7. Backend Prediction Workflow

When the user clicks **Submit**, the following steps occur:

1. **Input Collection**  
   User-provided values are collected from number fields and dropdowns.

2. **Categorical Encoding**  
   Each categorical input is transformed using the pre-loaded `LabelEncoder` objects from `encoders.pkl`.

3. **Numerical Scaling**  
   Duration and days_left values are normalized using `StandardScaler` from `scaler.pkl`.

4. **Feature Alignment**  
   The final ordered feature vector is built using the structure defined in `features.pkl`.

5. **Model Prediction**  
   The vector is passed into `flight_model.pkl`, which outputs a **continuous prediction** — the estimated ticket price.

6. **Result Display**  
   The predicted price is shown on the UI using an attractive, highlighted layout.

---

## 🚀 8. Deliverables & Deployment-Ready System

The Streamlit app ensures:
- **Zero retraining** (all processing logic uses pre-saved model assets)  
- **High accuracy** (thanks to preprocessing consistency)  
- **Fast predictions** (the model loads instantly in memory)  
- **End-user usability** (simple UI with clean input flow)  

This completes the **Model Training + Backend Integration + Streamlit Frontend Deployment** pipeline, resulting in a fully functional Flight Price Prediction system ready for real-world use.

