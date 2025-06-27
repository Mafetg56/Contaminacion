import streamlit as st
import pandas as pd
import joblib
import os

# Function to preprocess the data
def preprocess_data(df, scaler):
df = df.drop(['Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)', 'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)', 'Precipitación (mm)', 'Humedad Relativa 10m (%)'], axis=1)

# Apply Scaling
df['Velocidad del Viento (m/s)'] = scaler.transform(df[['Velocidad del Viento (m/s)']]) 
df['Dirección del viento (Grados)'] = scaler.transform(df[['Dirección del viento (Grados)']]) 
df['Presión atmosférica (mm Hg)'] = scaler.transform(df[['Presión atmosférica (mm Hg)']]) 
df['Radiación Solar Global (W/m2)'] = scaler.transform(df[['Radiación Solar Global (W/m2)']]) 

return df



    try:
        models = {
            'Linear Regression': joblib.load('best_linear_regression_model.pkl'),
            'KNN Regressor': joblib.load('best_knn_model.pkl'),
            'SVR': joblib.load('best_svm_model.pkl'),
            'Lasso Regression': joblib.load('best_lasso_model.pkl'),
            'Decision Tree Regressor': joblib.load('best_decision_tree_model.pkl'),
            'Voting Regressor': joblib.load('best_voting_regressor_model.pkl'),
            'Random Forest Regressor': joblib.load('best_random_forest_model.pkl'),
            'Gradient Boosting Regressor': joblib.load('best_gradient_boosting_model.pkl')
        }
  except FileNotFoundError:
    st.error("Error: Model files not found. Make sure 'saved_models' directory with required files exists.")
    st.stop()


# Streamlit App Title
st.title("Predicción de Contaminación del aire en PM10 (ug/m3)")

st.write("""
Esta aplicación predice la contaminación del aire basado en los datos metereológicos.
""")

# File uploader for the user to upload their data
uploaded_file = st.file_uploader("Sube tu archivo Excel (solo .xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded Excel file into a pandas DataFrame
        df = pd.read_excel(uploaded_file)

        st.subheader("Datos cargados:")
        st.write(df.head())
        
 # Preprocess the data
        st.subheader("Datos preprocesados:")
        processed_df = preprocess_data(df.copy(), scaler)
        st.write(processed_df.head())




# Make prediction
if st.sidebar.button("Predict"):
    try:
        prediction = selected_model.predict(input_scaled)
        st.subheader(f"Prediction using {selected_model_name}:")
        st.write(f"Predicted PM10 Concentration: {prediction[0]:.4f} ug/m3")

        # Optional: Display feature importances if available (e.g., for tree-based models)
        if hasattr(selected_model, 'feature_importances_'):
            st.subheader(f"Feature Importances ({selected_model_name})")
            importances = pd.Series(selected_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            importances.plot(kind='bar', ax=ax)
            ax.set_title(f'Feature Importances for {selected_model_name}')
            ax.set_ylabel('Importance')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


st.subheader("Model Performance (from notebook)")
st.write("Below are the evaluation metrics for the trained models on the test set from the original analysis:")


