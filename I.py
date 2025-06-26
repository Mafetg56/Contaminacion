import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import base64 # For download links
# No ydata-profiling as it's not designed for real-time Streamlit use.

# --- Helper Function for Download Links ---
def get_download_link(file_path, file_label):
    with open(file_path, 'rb') as f:
        data = f.read()
    bin_file = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_file}" download="{file_path}">{file_label}</a>'
    return href

# --- Streamlit App ---

st.set_page_config(layout="wide")

st.title("Análisis y Modelado de Calidad del Aire")

# --- File Upload ---
st.header("1. Cargar Datos")
uploaded_file = st.file_uploader("Sube tu archivo Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("Datos cargados exitosamente:")
    st.dataframe(df.head())

    # --- Data Cleaning and Preparation ---
    st.header("2. Limpieza y Preparación de Datos")

    # Drop specified columns
    columns_to_drop = ['Fecha y Hora de Inicio (dd/MM/aaaa  HH:mm:ss)',
                       'Fecha y Hora de Finalización (dd/MM/aaaa  HH:mm:ss)',
                       'Precipitación (mm)']
    initial_columns_dropped = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=initial_columns_dropped, axis=1, errors='ignore')
    if initial_columns_dropped:
        st.write(f"Columnas eliminadas inicialmente: {', '.join(initial_columns_dropped)}")
        st.dataframe(df.head())

    # Drop 'Humedad Relativa 10m (%)' if it exists
    humidity_col_to_drop = 'Humedad Relativa 10m (%)'
    if humidity_col_to_drop in df.columns:
        df = df.drop([humidity_col_to_drop], axis=1, errors='ignore')
        st.write(f"Columna eliminada: '{humidity_col_to_drop}'")
        st.dataframe(df.head())
    else:
        st.write(f"La columna '{humidity_col_to_drop}' no se encontró en el archivo.")

    st.write("Información del DataFrame después de la limpieza:")
    st.write(df.info())


    # --- Basic Statistics ---
    st.header("3. Estadísticas Descriptivas")
    st.write(df.describe())

    # --- Data Visualization ---
    st.header("4. Visualización de Datos")

    st.subheader("Diagramas de Caja (Box Plots)")
    df_numeric = df.select_dtypes(include=np.number)
    if not df_numeric.empty:
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(data=df_numeric, ax=ax)
        ax.set_title('Diagramas de Caja de Variables Numéricas')
        ax.tick_params(axis='x', rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No hay columnas numéricas para mostrar diagramas de caja.")

    # Outlier Identification (Displaying summary, not the full records in Streamlit for brevity)
    st.subheader("Identificación de Outliers (Resumen)")
    outliers_summary = {}
    for col in df_numeric.columns:
        Q1 = df_numeric[col].quantile(0.25)
        Q3 = df_numeric[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        col_outliers_count = df_numeric[(df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)].shape[0]
        if col_outliers_count > 0:
            outliers_summary[col] = col_outliers_count
    if outliers_summary:
        st.write("Número de outliers detectados por columna (usando el método IQR):")
        st.write(outliers_summary)
    else:
        st.write("No se detectaron outliers significativos (usando el método IQR).")


    # --- Data Scaling (for modeling) ---
    st.header("5. Escalado de Datos")
    df_scaled_modeling = df.copy()

    # Identify columns to scale (all numerical except the target)
    target_variable = 'PM10 (ug/m3)\nCondición Estándar' # Define target here
    columns_to_scale_modeling = df_scaled_modeling.select_dtypes(include=np.number).columns.tolist()
    if target_variable in columns_to_scale_modeling:
        columns_to_scale_modeling.remove(target_variable)

    if columns_to_scale_modeling:
        scaler_modeling = StandardScaler()
        df_scaled_modeling[columns_to_scale_modeling] = scaler_modeling.fit_transform(df_scaled_modeling[columns_to_scale_modeling])
        st.write("Datos escalados para el modelado (StandardScaler aplicado a características):")
        st.dataframe(df_scaled_modeling.head())
        # Save the scaler for future predictions
        joblib.dump(scaler_modeling, 'standard_scaler.pkl')
        st.write("Scaler guardado como `standard_scaler.pkl`.")
    else:
        st.write("No hay características para escalar (solo la variable objetivo).")


    # --- PCA Analysis ---
    st.header("6. Análisis de Componentes Principales (PCA)")

    df_numeric_pca = df.select_dtypes(include=np.number).copy()
    if not df_numeric_pca.empty:
        # Standardize the data before applying PCA
        scaler_pca = StandardScaler()
        df_scaled_pca = scaler_pca.fit_transform(df_numeric_pca)

        # Apply PCA
        pca = PCA()
        principal_components = pca.fit_transform(df_scaled_pca)

        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance)

        st.subheader("Varianza Explicada por Componente Principal")
        pca_summary_df = pd.DataFrame({
            'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Explained Variance Ratio': explained_variance,
            'Cumulative Explained Variance': cumulative_explained_variance
        })
        st.dataframe(pca_summary_df)

        st.subheader("Gráfico de Varianza Explicada Acumulada")
        fig_cum_var, ax_cum_var = plt.subplots(figsize=(10, 6))
        ax_cum_var.plot(range(1, len(explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
        ax_cum_var.set_title('Varianza Explicada por Número de Componentes Principales')
        ax_cum_var.set_xlabel('Número de Componentes Principales')
        ax_cum_var.set_ylabel('Ratio de Varianza Explicada Acumulada')
        ax_cum_var.grid(True)
        st.pyplot(fig_cum_var)

        st.subheader("Gráfico de Varianza Explicada por Componente")
        fig_var, ax_var = plt.subplots(figsize=(10, 6))
        ax_var.bar(range(1, len(explained_variance) + 1), explained_variance)
        ax_var.set_title('Varianza Explicada por Componente Principal')
        ax_var.set_xlabel('Componente Principal')
        ax_var.set_ylabel('Ratio de Varianza Explicada')
        st.pyplot(fig_var)

        # Optional: Visualize the first two principal components
        if principal_components.shape[1] >= 2:
            st.subheader("PCA - Primeros Dos Componentes Principales")
            fig_pca, ax_pca = plt.subplots(figsize=(10, 8))
            ax_pca.scatter(principal_components[:, 0], principal_components[:, 1])
            ax_pca.set_title('PCA - Primeros Dos Componentes Principales')
            ax_pca.set_xlabel('Componente Principal 1')
            ax_pca.set_ylabel('Componente Principal 2')
            ax_pca.grid(True)
            st.pyplot(fig_pca)
    else:
        st.write("No hay columnas numéricas para realizar PCA.")


    # --- Model Training and Evaluation ---
    st.header("7. Entrenamiento y Evaluación de Modelos")

    # Define features (X) and target (y)
    # Ensure the target variable name matches exactly, including the newline if present
    if target_variable not in df.columns:
         # Attempt to find the correct target column if the expected name doesn't match
        matching_cols = [col for col in df.columns if 'PM10' in col and 'Condición Estándar' in col]
        if matching_cols:
            target_variable = matching_cols[0]
            st.warning(f"Using found target column: '{target_variable}'")
        else:
            st.error(f"Target variable '{target_variable}' not found in DataFrame columns: {df.columns.tolist()}")
            st.stop() # Stop execution if target column is not found


    numerical_cols_model = df.select_dtypes(include=np.number).columns
    feature_cols = [col for col in numerical_cols_model if col != target_variable]

    if not feature_cols:
         st.error("No features available to train the model after dropping columns.")
         st.stop()

    X = df[feature_cols]
    y = df[target_variable]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write(f"Dimensiones de los datos de entrenamiento: X={X_train.shape}, y={y_train.shape}")
    st.write(f"Dimensiones de los datos de prueba: X={X_test.shape}, y={y_test.shape}")

    # Dictionary to store evaluation metrics for different models
    evaluation_metrics = {}

    # Function to evaluate a model
    def evaluate_model(model, X_test, y_test, model_name):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Calculate MAPE
        y_test_non_zero = y_test[y_test != 0]
        y_pred_non_zero = y_pred[y_test != 0]
        mape = np.mean(np.abs((y_test_non_zero - y_pred_non_zero) / y_test_non_zero)) * 100 if len(y_test_non_zero) > 0 else float('inf')

        metrics = {
            'MAE': mae,
            'MAPE (%)': mape,
            'RMSE': rmse,
            'R2': r2
        }
        evaluation_metrics[model_name] = metrics
        return metrics, y_pred

    # List of models to train and evaluate
    models = {
        "Regresión Lineal": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(),
        "SVR": SVR(),
        "Lasso Regressor": Lasso(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
    }

    # Add Voting Regressor using default instances for now
    # Note: For optimal performance, you'd tune individual models first and then the VotingRegressor
    # or tune the VotingRegressor with base models directly.
    models["Voting Regressor"] = VotingRegressor(
        estimators=[
            ('lr', LinearRegression()),
            ('knn', KNeighborsRegressor()),
            ('svm', SVR()),
            ('lasso', Lasso()),
            ('dt', DecisionTreeRegressor(random_state=42))
        ]
    )

    trained_models = {}

    st.subheader("Resultados de la Evaluación del Modelo (con parámetros por defecto)")

    results_data = []

    for model_name, model in models.items():
        st.write(f"--- Entrenamiento y Evaluación de: {model_name} ---")
        try:
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            metrics, y_pred = evaluate_model(model, X_test, y_test, model_name)

            st.write(f"Métricas de Evaluación para {model_name}:")
            st.write(f"  MAE: {metrics['MAE']:.4f}")
            st.write(f"  MAPE: {metrics['MAPE (%)']:.4f}%")
            st.write(f"  RMSE: {metrics['RMSE']:.4f}")
            st.write(f"  R2: {metrics['R2']:.4f}")

            results_data.append({
                'Modelo': model_name,
                'MAE': metrics['MAE'],
                'MAPE (%)': metrics['MAPE (%)'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            })

            # Optional: Plot predictions vs actual
            fig_pred_actual, ax_pred_actual = plt.subplots(figsize=(10, 6))
            ax_pred_actual.scatter(y_test, y_pred, alpha=0.5)
            ax_pred_actual.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax_pred_actual.set_xlabel("Valores Reales")
            ax_pred_actual.set_ylabel("Predicciones")
            ax_pred_actual.set_title(f"Valores Reales vs. Predicciones ({model_name})")
            st.pyplot(fig_pred_actual)

            # Optional: Save model
            model_filename = model_name.lower().replace(" ", "_").replace("-", "") + "_model.pkl"
            joblib.dump(model, model_filename)
            st.write(f"Modelo guardado como `{model_filename}`.")

        except Exception as e:
            st.error(f"Error entrenando o evaluando {model_name}: {e}")
            results_data.append({
                'Modelo': model_name,
                'MAE': 'Error',
                'MAPE (%)': 'Error',
                'RMSE': 'Error',
                'R2': 'Error'
            })

    st.subheader("Resumen de Métricas de Evaluación")
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df.set_index('Modelo'))

    # --- Optional: Hyperparameter Tuning (Simplified for Streamlit) ---
    st.header("8. Tuning de Hiperparámetros (Opcional)")
    st.write("Nota: El tuning de hiperparámetros con GridSearchCV puede ser lento. Aquí solo mostramos los resultados si los modelos tunados ya existen.")

    # Check if tuned models exist and display their results
    tuned_models_info = {
        'Best Linear Regression': 'best_linear_regression_model.pkl',
        'Best KNN Regressor': 'best_knn_model.pkl',
        'Best SVR': 'best_svm_model.pkl',
        'Best Lasso Regressor': 'best_lasso_model.pkl',
        'Best Decision Tree': 'best_decision_tree_model.pkl',
        'Best Voting Regressor': 'best_voting_regressor_model.pkl',
        'Best Random Forest': 'best_random_forest_model.pkl',
        'Best Gradient Boosting': 'best_gradient_boosting_model.pkl'
    }

    tuned_results_data = []

    for tuned_name, tuned_filename in tuned_models_info.items():
        try:
            tuned_model = joblib.load(tuned_filename)
            st.write(f"--- Evaluación de: {tuned_name} (Modelo Tunado) ---")
            metrics, y_pred_tuned = evaluate_model(tuned_model, X_test, y_test, tuned_name)
            st.write(f"Métricas de Evaluación para {tuned_name}:")
            st.write(f"  MAE: {metrics['MAE']:.4f}")
            st.write(f"  MAPE: {metrics['MAPE (%)']:.4f}%")
            st.write(f"  RMSE: {metrics['RMSE']:.4f}")
            st.write(f"  R2: {metrics['R2']:.4f}")

            tuned_results_data.append({
                'Modelo Tunado': tuned_name,
                'MAE': metrics['MAE'],
                'MAPE (%)': metrics['MAPE (%)'],
                'RMSE': metrics['RMSE'],
                'R2': metrics['R2']
            })

        except FileNotFoundError:
            st.write(f"Modelo tunado '{tuned_filename}' no encontrado. Ejecute el script original para generarlos.")
        except Exception as e:
            st.error(f"Error cargando o evaluando el modelo tunado {tuned_name}: {e}")
            tuned_results_data.append({
                'Modelo Tunado': tuned_name,
                'MAE': 'Error',
                'MAPE (%)': 'Error',
                'RMSE': 'Error',
                'R2': 'Error'
            })

    if tuned_results_data:
        st.subheader("Resumen de Métricas de Evaluación (Modelos Tunados)")
        tuned_results_df = pd.DataFrame(tuned_results_data)
        st.dataframe(tuned_results_df.set_index('Modelo Tunado'))


    # --- Make Predictions (Simple Example) ---
    st.header("9. Realizar Predicciones")
    st.write("Puedes usar los modelos entrenados para hacer predicciones sobre nuevos datos.")

    # Select a model to use for prediction
    model_choice = st.selectbox("Selecciona un modelo para predicción:", list(trained_models.keys()))

    if model_choice:
        selected_model = trained_models[model_choice]

        st.write(f"Modelo seleccionado: **{model_choice}**")
        st.write("Ingresa los valores de las características para hacer una predicción.")

        # Create input fields for each feature
        input_values = {}
        for feature in feature_cols:
            input_values[feature] = st.number_input(f"Valor para '{feature}'", value=float(X_train[feature].mean())) # Default to mean value

        if st.button("Predecir"):
            # Prepare the input data
            input_df = pd.DataFrame([input_values])

            # Apply the same scaling used during training
            # Need to ensure the columns are in the same order as during training/scaling
            input_df_scaled = input_df.copy()
            # Use the feature_cols list to ensure order and select columns for scaling
            cols_to_scale_input = [col for col in feature_cols if col in columns_to_scale_modeling]
            if cols_to_scale_input:
                 # Load the scaler fitted on training features
                 try:
                     loaded_scaler = joblib.load('standard_scaler.pkl')
                     input_df_scaled[cols_to_scale_input] = loaded_scaler.transform(input_df_scaled[cols_to_scale_input])
                 except FileNotFoundError:
                     st.error("Error: Scaler no encontrado. Por favor, asegura que el archivo 'standard_scaler.pkl' existe.")
                     st.stop()
                 except Exception as e:
                     st.error(f"Error aplicando el scaler: {e}")
                     st.stop()

            # Ensure the input DataFrame has the exact same columns and order as X_train
            # This is crucial for models like Linear Regression, Tree-based models, etc.
            # If the input data might be missing columns (which shouldn't happen with fixed inputs here),
            # you might need to add them with default values (e.g., 0 or mean).
            input_df_final = input_df_scaled[feature_cols]


            try:
                prediction = selected_model.predict(input_df_final)[0]
                st.subheader("Resultado de la Predicción:")
                st.write(f"La predicción para {target_variable} es: **{prediction:.4f}**")
            except Exception as e:
                st.error(f"Error al hacer la predicción: {e}")

else:
    st.info("Por favor, carga un archivo Excel para comenzar el análisis.")
