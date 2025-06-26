import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data (adjust path as needed)
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Mediciones_Calidad_Aire_La_Candelaria_Organizado.xlsx")
        # Apply the same data cleaning steps as in the notebook
        df = df.drop(['Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)', 'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)', 'Precipitación (mm)'], axis=1)
        if 'Humedad Relativa 10m (%)' in df.columns:
             df = df.drop(['Humedad Relativa 10m (%)'], axis=1)
        return df
    except FileNotFoundError:
        st.error("Error: 'Mediciones_Calidad_Aire_La_Candelaria_Organizado.xlsx' not found.")
        st.stop() # Stop the app if the file is not found

# Function to load trained models and scaler
@st.cache_resource
def load_models():
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
        scaler = joblib.load('standard_scaler.pkl')
        return models, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading models or scaler: {e}. Please ensure all .pkl files are in the same directory.")
        st.stop() # Stop the app if files are not found
    except Exception as e:
        st.error(f"An error occurred while loading models: {e}")
        st.stop()

# Load data, models, and scaler
df = load_data()
models, scaler = load_models()

# Define the target variable and features based on the notebook
target_variable = 'PM10 (ug/m3)\nCondición Estándar'
if target_variable in df.columns:
    numerical_cols = df.select_dtypes(include=np.number).columns
    feature_cols = [col for col in numerical_cols if col != target_variable]
else:
     st.error(f"Target variable '{target_variable}' not found in the dataset columns.")
     st.stop()

# Check if features are available
if not feature_cols:
    st.error("No suitable numerical features found in the dataset after cleaning.")
    st.stop()


# Streamlit App Layout
st.title("Air Quality Prediction App (La Candelaria)")

st.write("""
This app predicts PM10 concentration based on selected atmospheric parameters
using various regression models.
""")

st.subheader("Dataset Overview")
st.write("First few rows of the processed dataset:")
st.dataframe(df.head())

st.write("Dataset Information:")
buffer = pd.io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)


st.subheader("Predict PM10 Concentration")

# Sidebar for user input and model selection
st.sidebar.header("Input Parameters")

# Create input fields for each feature
input_data = {}
st.sidebar.write("Enter values for the following parameters:")

for col in feature_cols:
    # Use a number_input for numerical features
    min_val = df[col].min() if not pd.isna(df[col].min()) else 0.0
    max_val = df[col].max() if not pd.isna(df[col].max()) else 100.0
    mean_val = df[col].mean() if not pd.isna(df[col].mean()) else 50.0
    input_data[col] = st.sidebar.number_input(
        f"{col}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(mean_val) # Default to mean value
    )

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Scale the input data using the fitted scaler
try:
    input_scaled = scaler.transform(input_df)
except Exception as e:
    st.error(f"Error scaling input data: {e}")
    st.stop()


st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a regression model:", list(models.keys()))

# Get the selected model
selected_model = models[selected_model_name]

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

# You can manually display the metrics obtained in the notebook
# Or load them if you saved them to a file

st.write("Linear Regression:")
st.write(f"- MAE: {mae_best_lr:.4f}")
st.write(f"- MAPE: {mape_best_lr:.4f}%")
st.write(f"- RMSE: {rmse_best_lr:.4f}")
st.write(f"- R2: {r2_best_lr:.4f}")

st.write("KNN Regressor:")
st.write(f"- MAE: {mae_best_knn:.4f}")
st.write(f"- MAPE: {mape_best_knn:.4f}%")
st.write(f"- RMSE: {rmse_best_knn:.4f}")
st.write(f"- R2: {r2_best_knn:.4f}")

st.write("SVR:")
st.write(f"- MAE: {mae_best_svm:.4f}")
st.write(f"- MAPE: {mape_best_svm:.4f}%")
st.write(f"- RMSE: {rmse_best_svm:.4f}")
st.write(f"- R2: {r2_best_svm:.4f}")

st.write("Lasso Regression:")
st.write(f"- MAE: {mae_best_lasso:.4f}")
st.write(f"- MAPE: {mape_best_lasso:.4f}%")
st.write(f"- RMSE: {rmse_best_lasso:.4f}")
st.write(f"- R2: {r2_best_lasso:.4f}")

st.write("Decision Tree Regressor:")
st.write(f"- MAE: {mae_best_dt:.4f}")
st.write(f"- MAPE: {mape_best_dt:.4f}%")
st.write(f"- RMSE: {rmse_best_dt:.4f}")
st.write(f"- R2: {r2_best_dt:.4f}")

st.write("Voting Regressor:")
st.write(f"- MAE: {mae_best_voting:.4f}")
st.write(f"- MAPE: {mape_best_voting:.4f}%")
st.write(f"- RMSE: {rmse_best_voting:.4f}")
st.write(f"- R2: {r2_best_voting:.4f}")

st.write("Random Forest Regressor:")
st.write(f"- MAE: {mae_best_rf:.4f}")
st.write(f"- MAPE: {mape_best_rf:.4f}%")
st.write(f"- RMSE: {rmse_best_rf:.4f}")
st.write(f"- R2: {r2_best_rf:.4f}")

st.write("Gradient Boosting Regressor:")
st.write(f"- MAE: {mae_best_gbr:.4f}")
st.write(f"- MAPE: {mape_best_gbr:.4f}%")
st.write(f"- RMSE: {rmse_best_gbr:.4f}")
st.write(f"- R2: {r2_best_gbr:.4f}")


st.subheader("Data Visualization (from notebook)")
st.write("Some visualizations from the data preparation step:")

# Recreate and display the boxplot if needed
st.write("Box Plots of Numerical Variables:")
df_numeric_for_viz = df.select_dtypes(include=np.number).copy()
if target_variable in df_numeric_for_viz.columns:
    df_numeric_for_viz = df_numeric_for_viz.drop(columns=[target_variable]) # Exclude target for boxplot if desired

if not df_numeric_for_viz.empty:
    fig_box, ax_box = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=df_numeric_for_viz, ax=ax_box)
    ax_box.set_title('Box Plots of Numerical Variables')
    ax_box.tick_params(axis='x', rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig_box)
else:
    st.write("No numerical columns available for boxplot after excluding the target variable.")

# You could also add visualizations for PCA results, feature importances (general), etc.
