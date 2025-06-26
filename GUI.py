# prompt: Haz todo el despliegue anterior en streamlit
import streamlit as st
import pandas as pd
import joblib
import os

# Function to preprocess the data
def preprocess_data(df, encoder, scaler):
    df['Felder'] = df['Felder'].astype('category')
    df = df.drop(['ID', 'Año - Semestre', 'Nota_final'], axis=1)

    # Apply One-Hot Encoding
    encoded_felder = encoder.transform(df[['Felder']])
    encoded_felder_df = pd.DataFrame(encoded_felder.toarray(), columns=encoder.get_feature_names_out(['Felder']))
    df = pd.concat([df.drop('Felder', axis=1), encoded_felder_df], axis=1)

    # Apply Scaling
    df['Examen_admisión'] = scaler.transform(df[['Examen_admisión']]) # Use transform after fit_transform was called during training

    return df

# Load the pre-trained models and transformers
# Assuming your models and transformers are in a directory named 'saved_models'
# in the same directory as your Streamlit app or accessible via a path.
try:
    encoder = joblib.load('saved_models/one_hot_encoder.pkl')
    scaler = joblib.load('saved_models/standard_scaler.pkl')
    rf_model = joblib.load('saved_models/random_forest_regressor_model.pkl')
    lr_model = joblib.load('saved_models/linear_regressor_model.pkl')
    mlp_model = joblib.load('saved_models/mlp_regressor_model.pkl')
except FileNotFoundError:
    st.error("Error: Model files not found. Make sure 'saved_models' directory with required files exists.")
    st.stop()

# Streamlit App Title
st.title("Predicción de Nota Final de Curso")

st.write("""
Esta aplicación predice la nota final de un curso basado en los datos del estudiante.
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
        processed_df = preprocess_data(df.copy(), encoder, scaler)
        st.write(processed_df.head())

        # Make predictions using different models
        st.subheader("Predicciones:")

        # Random Forest Regressor
        rf_predictions = rf_model.predict(processed_df)
        df['Predicted_Nota_RF'] = rf_predictions
        st.write("Predicciones (Random Forest Regressor):")
        st.write(df[['ID', 'Predicted_Nota_RF']].head())

        # Linear Regressor
        lr_predictions = lr_model.predict(processed_df)
        df['Predicted_Nota_LR'] = lr_predictions
        st.write("Predicciones (Linear Regressor):")
        st.write(df[['ID', 'Predicted_Nota_LR']].head())

        # MLP Regressor
        mlp_predictions = mlp_model.predict(processed_df)
        df['Predicted_Nota_MLP'] = mlp_predictions
        st.write("Predicciones (MLP Regressor):")
        st.write(df[['ID', 'Predicted_Nota_MLP']].head())

        st.subheader("Resultados Completos:")
        st.write(df[['ID', 'Examen_admisión', 'Felder', 'Predicted_Nota_RF', 'Predicted_Nota_LR', 'Predicted_Nota_MLP']])

    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")
