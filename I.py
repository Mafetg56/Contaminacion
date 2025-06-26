import pandas as pd
import numpy as np
df = pd.read_excel(uploaded_file)

# Display the raw data (optional)
st.subheader("Datos Originales")
st.write(df.head())

# --- Data Preprocessing ---
# We need to apply the same preprocessing steps as used during training.
# Based on the provided Colab notebook code:
# 1. Drop specific columns: 'Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)',
#    'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)', 'Precipitación (mm)'
# 2. Drop 'Humedad Relativa 10m (%)'
# 3. Scale numerical features (all except the target 'PM10 (ug/m3)\nCondición Estándar')

# Define the target variable name (careful with newline characters)
target_variable = 'PM10 (ug/m3)\nCondición Estándar'

# Define columns to drop based on the Colab notebook
cols_to_drop = [
            'Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)',
            'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)',
            'Precipitación (mm)',
            'Humedad Relativa 10m (%)' # This was dropped after mutual information analysis
        ]

# Drop specified columns
# Use errors='ignore' in case a column is not present in the uploaded file
        df_processed = df.drop(columns=cols_to_drop, errors='ignore')

# Identify numerical columns for scaling (excluding the target if present)
        numerical_cols_processed = df_processed.select_dtypes(include=np.number).columns.tolist()
        feature_cols_processed = [col for col in numerical_cols_processed if col != target_variable]

# Check if the target variable is in the processed DataFrame.
# If not, it means the uploaded file is likely for prediction and doesn't contain the target.
# In this case, we only need to preprocess the features.
        if target_variable in df_processed.columns:
            X_new = df_processed[feature_cols_processed]
            # Assuming the user uploaded data with the target for potential comparison or evaluation
            # If you strictly only expect data *for* prediction (without the target), remove the y_new part.
            y_new = df_processed[target_variable]
        else:
             X_new = df_processed[feature_cols_processed]
             y_new = None # No target variable in the uploaded file


# Load the scaler fitted on the training data features
# Assuming 'standard_scaler.pkl' was saved from the training notebook and is accessible.
        try:
            scaler_for_features = joblib.load('standard_scaler.pkl')
            # Apply scaling to the new feature data
            X_new_scaled = scaler_for_features.transform(X_new)
            # Convert scaled features back to a DataFrame for easier handling if needed
            X_new_scaled_df = pd.DataFrame(X_new_scaled, columns=X_new.columns, index=X_new.index)

            st.subheader("Datos Preprocesados (Características Escaladas)")
            st.write(X_new_scaled_df.head())

        except FileNotFoundError:
            st.error("Error: 'standard_scaler.pkl' not found. Make sure the scaler file is available.")
            st.stop()
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            st.stop()


# --- Model Selection and Prediction ---
        st.subheader("Selecciona un Modelo para la Predicción")

# List available trained models
# Assuming the best models were saved with specific names in the Colab notebook
        model_files = {
            "Linear Regression": 'best_linear_regression_model.pkl',
            "KNN Regressor": 'best_knn_model.pkl',
            "SVR Regressor": 'best_svm_model.pkl',
            "Lasso Regressor": 'best_lasso_model.pkl',
            "Decision Tree Regressor": 'best_decision_tree_model.pkl',
            "Voting Regressor": 'best_voting_regressor_model.pkl',
            "Random Forest Regressor": 'best_random_forest_model.pkl',
            "Gradient Boosting Regressor": 'best_gradient_boosting_model.pkl',
        }

        selected_model_name = st.selectbox(
            "Elige un modelo:",
            list(model_files.keys())
        )

        model_file_path = model_files[selected_model_name]

        try:
            # Load the selected model
            model = joblib.load(model_file_path)

            st.write(f"Modelo seleccionado: **{selected_model_name}** cargado exitosamente.")

            # Make predictions using the loaded model
            predictions = model.predict(X_new_scaled_df) # Predict using the scaled features

            st.subheader(f"Predicciones con {selected_model_name}")
            # Display the predictions
            predictions_df = pd.DataFrame(predictions, columns=["Predicción PM10"])
            st.write(predictions_df)

            # Optionally, if the uploaded data had the target, show a comparison
            if y_new is not None:
                st.subheader("Comparación de Valores Reales y Predicciones")
                comparison_df = pd.DataFrame({'Real PM10': y_new, 'Predicción PM10': predictions})
                st.write(comparison_df)

                # Optional: Display evaluation metrics on this new data (be cautious,
                # these metrics evaluate the model's performance on the *new* data, not the test set it was tuned on).
                st.subheader("Métricas de Evaluación en los Datos Cargados")
                mae_new = mean_absolute_error(y_new, predictions)
                rmse_new = np.sqrt(mean_squared_error(y_new, predictions))
                r2_new = r2_score(y_new, predictions)

                y_new_non_zero = y_new[y_new != 0]
                predictions_non_zero = predictions[y_new != 0]
                mape_new = np.mean(np.abs((y_new_non_zero - predictions_non_zero) / y_new_non_zero)) * 100 if len(y_new_non_zero) > 0 else float('inf')

                st.write(f"Mean Absolute Error (MAE): {mae_new:.4f}")
                st.write(f"Mean Absolute Percentage Error (MAPE): {mape_new:.4f}%")
                st.write(f"Root Mean Squared Error (RMSE): {rmse_new:.4f}")
                st.write(f"R-squared (R2): {r2_new:.4f}")


        except FileNotFoundError:
            st.error(f"Error: El archivo del modelo '{model_file_path}' no fue encontrado.")
        except Exception as e:
            st.error(f"Error al cargar o usar el modelo: {e}")

    except Exception as e:
        st.error(f"Error al leer el archivo Excel: {e}")



  else:
    st.warning("Data splitting or scaling failed. Cannot proceed with model training/evaluation.")

else:
  st.info("Please upload an Excel file to get started.")
