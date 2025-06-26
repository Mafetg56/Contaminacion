import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Air Quality Prediction App")

st.title("Air Quality Prediction Dashboard")

# --- Helper Functions ---

@st.cache_data
def load_data(uploaded_file):
  """Loads data from an uploaded file."""
  if uploaded_file is not None:
    try:
      df = pd.read_excel(uploaded_file)
      # Perform initial cleaning and column dropping as in the notebook
      df = df.drop(['Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)',
                    'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)',
                    'Precipitación (mm)'], axis=1, errors='ignore')
      if 'Humedad Relativa 10m (%)' in df.columns:
        df = df.drop(['Humedad Relativa 10m (%)'], axis=1)
      return df
    except Exception as e:
      st.error(f"Error loading file: {e}")
      return None
  return None

@st.cache_resource
def load_model(model_path):
  """Loads a trained model."""
  try:
    model = joblib.load(model_path)
    return model
  except Exception as e:
    st.error(f"Error loading model: {e}")
    return None

@st.cache_resource
def load_scaler(scaler_path):
  """Loads a trained scaler."""
  try:
    scaler = joblib.load(scaler_path)
    return scaler
  except Exception as e:
    st.error(f"Error loading scaler: {e}")
    return None

def evaluate_model(model, X_test, y_test):
  """Evaluates a given model and returns metrics."""
  y_pred = model.predict(X_test)

  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mean_squared_error(y_test, y_pred))
  r2 = r2_score(y_test, y_pred)

  # Calculate MAPE, handling division by zero
  y_test_non_zero = y_test[y_test != 0]
  y_pred_non_zero = y_pred[y_test != 0]
  mape = np.mean(np.abs((y_test_non_zero - y_pred_non_zero) / y_test_non_zero)) * 100 if len(y_test_non_zero) > 0 else float('inf')

  return mae, rmse, r2, mape, y_pred

# --- File Upload ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your Excel data file", type=["xlsx"])

df = load_data(uploaded_file)

if df is not None:
  st.sidebar.success("File uploaded and processed successfully!")
  st.header("Data Overview")
  st.write("First 5 rows of the processed data:")
  st.dataframe(df.head())
  st.write("Data Info:")
  st.text(df.info())

  # --- Data Preprocessing Steps (adapted for Streamlit) ---

  st.header("Data Preprocessing")

  # Display columns after initial drop
  st.subheader("Columns after initial processing:")
  st.write(df.columns.tolist())


  # Outlier Visualization (Box Plot)
  st.subheader("Outlier Visualization (Box Plots)")
  df_numeric = df.select_dtypes(include=np.number)
  if not df_numeric.empty:
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.boxplot(data=df_numeric, ax=ax)
    ax.set_title('Box Plots of Numerical Variables')
    ax.set_xticks(range(len(df_numeric.columns)))
    ax.set_xticklabels(df_numeric.columns, rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
  else:
    st.write("No numerical columns to display box plots.")

  # Outlier Identification (Display only)
  st.subheader("Identified Outliers")
  outliers = {}
  for col in df_numeric.columns:
      Q1 = df_numeric[col].quantile(0.25)
      Q3 = df_numeric[col].quantile(0.75)
      IQR = Q3 - Q1
      lower_bound = Q1 - 1.5 * IQR
      upper_bound = Q3 + 1.5 * IQR
      col_outliers = df_numeric[(df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)]
      if not col_outliers.empty:
          outliers[col] = col_outliers

  if outliers:
      st.write("Outlier Records (showing first 5 for each column):")
      for col, outlier_df in outliers.items():
          st.write(f"**Outliers for {col}:**")
          st.dataframe(outlier_df.head()) # Show only the first few outlier rows
  else:
      st.write("No significant outliers detected based on the IQR method.")


  # Feature Scaling
  st.subheader("Feature Scaling (using StandardScaler)")
  # Assuming the target variable is 'PM10 (ug/m3)\nCondición Estándar'
  target_variable = 'PM10 (ug/m3)\nCondición Estándar'

  if target_variable in df.columns:
    numerical_cols = df.select_dtypes(include=np.number).columns
    feature_cols = [col for col in numerical_cols if col != target_variable]

    if feature_cols:
      X = df[feature_cols]
      y = df[target_variable]

      # Split the data before scaling to prevent data leakage
      from sklearn.model_selection import train_test_split
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

      # Initialize and fit the scaler on the training data
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)

      st.write("Features have been scaled using StandardScaler.")
      st.write("Shape of scaled training data:", X_train_scaled.shape)
      st.write("Shape of scaled test data:", X_test_scaled.shape)

      # Save the scaler for later use in prediction
      joblib.dump(scaler, 'standard_scaler.pkl')
      st.success("Standard scaler trained on training features and saved.")

      # PCA Analysis (Optional but included based on notebook)
      st.subheader("Principal Component Analysis (PCA)")
      # Apply PCA on the scaled training data
      pca = PCA()
      principal_components_train = pca.fit_transform(X_train_scaled)

      explained_variance = pca.explained_variance_ratio_
      cumulative_explained_variance = np.cumsum(explained_variance)

      st.write("Explained Variance Ratio by Principal Component:")
      pca_summary_df = pd.DataFrame({
          'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
          'Explained Variance Ratio': explained_variance,
          'Cumulative Explained Variance': cumulative_explained_variance
      })
      st.dataframe(pca_summary_df)

      # Plot explained variance
      fig_pca_cum, ax_pca_cum = plt.subplots(figsize=(10, 6))
      ax_pca_cum.plot(range(1, len(explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
      ax_pca_cum.set_title('Explained Variance by Number of Principal Components')
      ax_pca_cum.set_xlabel('Number of Principal Components')
      ax_pca_cum.set_ylabel('Cumulative Explained Variance Ratio')
      ax_pca_cum.grid(True)
      st.pyplot(fig_pca_cum)

      fig_pca_ind, ax_pca_ind = plt.subplots(figsize=(10, 6))
      ax_pca_ind.bar(range(1, len(explained_variance) + 1), explained_variance)
      ax_pca_ind.set_title('Explained Variance per Principal Component')
      ax_pca_ind.set_xlabel('Principal Component')
      ax_pca_ind.set_ylabel('Explained Variance Ratio')
      st.pyplot(fig_pca_ind)

      # Decide on the number of components for potential dimensionality reduction if needed
      # For this Streamlit app, we will use the original scaled features for model training
      # but the PCA analysis helps understand the data structure.

    else:
      st.warning("No numerical features found excluding the target variable.")
      X_train_scaled = None
      X_test_scaled = None
      y_train = None
      y_test = None

  else:
    st.error(f"Target variable '{target_variable}' not found in the DataFrame.")
    X_train_scaled = None
    X_test_scaled = None
    y_train = None
    y_test = None


  # --- Model Training and Evaluation ---
  if X_train_scaled is not None and X_test_scaled is not None:
    st.header("Model Training and Evaluation")

    # Placeholder for model loading/training based on your saved models
    st.write("Loading pre-trained models...")

    models = {
        "Linear Regression": load_model('best_linear_regression_model.pkl'),
        "KNN Regressor": load_model('best_knn_model.pkl'),
        "SVR": load_model('best_svm_model.pkl'),
        "Lasso Regression": load_model('best_lasso_model.pkl'),
        "Decision Tree Regressor": load_model('best_decision_tree_model.pkl'),
        "Voting Regressor": load_model('best_voting_regressor_model.pkl'),
        "Random Forest Regressor": load_model('best_random_forest_model.pkl'),
        "Gradient Boosting Regressor": load_model('best_gradient_boosting_model.pkl'),
    }

    evaluation_results = {}
    predictions = {}

    # Evaluate each loaded model
    for model_name, model in models.items():
      if model:
        st.subheader(f"Evaluating {model_name}")
        mae, rmse, r2, mape, y_pred = evaluate_model(model, X_test_scaled, y_test) # Evaluate on scaled test data
        evaluation_results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        predictions[model_name] = y_pred
        st.write(f"**{model_name} Metrics:**")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"R2: {r2:.4f}")
        st.write(f"MAPE: {mape:.4f}%")
      else:
        st.warning(f"Could not load model: {model_name}")

    # Display summary of evaluation metrics
    if evaluation_results:
      st.header("Model Comparison")
      eval_df = pd.DataFrame(evaluation_results).T
      st.dataframe(eval_df.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE'], axis=0).highlight_max(subset=['R2'], axis=0))

      # Optional: Plot actual vs predicted for a chosen model
      st.header("Actual vs. Predicted Values")
      model_choice = st.selectbox("Select a model to visualize predictions:", list(predictions.keys()))

      if model_choice:
        y_test_array = y_test.values # Ensure y_test is a numpy array
        y_pred_array = predictions[model_choice]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test_array, y_pred_array, alpha=0.3)
        ax.plot([y_test_array.min(), y_test_array.max()], [y_test_array.min(), y_test_array.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"Actual vs. Predicted Values ({model_choice})")
        st.pyplot(fig)

    else:
      st.warning("No models were loaded or evaluated.")

    # --- Make Predictions on New Data ---
    st.header("Make Predictions on New Data")
    st.write("Enter values for the features to get a prediction.")

    # Create input fields for each feature
    new_data = {}
    if feature_cols:
      for col in feature_cols:
        # Use appropriate input widgets based on expected data type if possible
        new_data[col] = st.number_input(f"Enter value for {col}", value=0.0, step=0.01)

      if st.button("Predict"):
        # Create a DataFrame from the input data
        new_data_df = pd.DataFrame([new_data])

        # Load the trained scaler
        scaler = load_scaler('standard_scaler.pkl')

        if scaler:
            # Scale the new data using the loaded scaler
            new_data_scaled = scaler.transform(new_data_df)

            # Choose a model for prediction (e.g., the best R2 model)
            if evaluation_results:
                best_model_name_for_pred = max(evaluation_results, key=lambda k: evaluation_results[k].get('R2', -float('inf')))
                model_for_prediction = models.get(best_model_name_for_pred)

                if model_for_prediction:
                    prediction = model_for_prediction.predict(new_data_scaled)
                    st.success(f"Predicted {target_variable} using {best_model_name_for_pred}: {prediction[0]:.4f}")
                else:
                    st.error("Could not load the best model for prediction.")
            else:
                 st.warning("No models were evaluated to select the best one for prediction.")
        else:
             st.error("Could not load the standard scaler. Cannot make predictions.")

    else:
        st.warning("No features available for prediction.")

  else:
    st.warning("Data splitting or scaling failed. Cannot proceed with model training/evaluation.")

else:
  st.info("Please upload an Excel file to get started.")
