import os
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

from utils.create_and_save_plot import plot_credit_score_distribution
from utils.calculate_probability_prediction import probability_to_score_v3
from utils.category_classification import credit_score_range_classification

# Cargar modelo y preprocesador
model = tf.keras.models.load_model("app/models/loan_model_2.h5")
preprocessor = joblib.load("app/models/preprocessor.pkl")


def calculate_dti(annual_income, total_monthly_debt):
    """Calcula el Debt-to-Income Ratio (DTI)"""
    if annual_income > 0:
        monthly_income = annual_income / 12
        dti = (total_monthly_debt / monthly_income) * 100
        return round(dti, 2)
    return 0


# Interfaz Streamlit
st.title("Predicción de Puntaje de Crédito")

st.sidebar.header("Ingrese los valores del préstamo")

# Nuevas variables de entrada con descripciones
annual_inc = st.sidebar.number_input(
    "Ingreso Anual (USD)", min_value=0.0, value=36000.0,
    help="(float) Ingreso anual en dólares antes de impuestos."
)
emp_length = st.sidebar.selectbox(
    "Tiempo en el Trabajo", ["< 1 year", "1-5 years", "6-10 years", "10+ years"],
    help="(categoría) Duración del empleo actual."
)
home_ownership = st.sidebar.selectbox(
    "Tipo de Propiedad", ["OWN", "MORTGAGE", "RENT"],
    help="(categoría) Tipo de propiedad del solicitante."
)
purpose = st.sidebar.selectbox(
    "Propósito del Préstamo", ["debt_consolidation", "credit_card", "home_improvement"],
    help="(categoría) Razón principal del préstamo."
)
zip_code = st.sidebar.number_input(
    "Código Postal", min_value=10000, max_value=99999, step=1, value=90210,
    help="(int) Código postal de residencia del solicitante."
)
open_acc = st.sidebar.number_input(
    "Cuentas Abiertas", min_value=0, step=1, value=5,
    help="(int) Número total de cuentas de crédito abiertas."
)
total_monthly_debt = st.sidebar.number_input(
    "Pagos Mensuales de Deuda (USD)", min_value=0.0, value=600.0,
    help="(float) Total de pagos mensuales de deuda (préstamos, tarjetas, hipotecas)."
)

# Calcular DTI
dti = calculate_dti(annual_inc, total_monthly_debt)
st.sidebar.write(f"DTI Calculado: {dti}%")

if st.sidebar.button("Predecir"):
    # Crear el DataFrame con los datos de entrada
    input_data = {
        "annual_inc": [annual_inc],
        "emp_length": [emp_length],
        "home_ownership": [home_ownership],
        "purpose": [purpose],
        "zip_code": [zip_code],
        "open_acc": [open_acc],
        "dti": [dti]
    }
    df = pd.DataFrame(input_data)

    # Preprocesar y predecir
    data_processed = preprocessor.transform(df)
    predictions = model.predict(data_processed).ravel()
    y_scores = joblib.load(os.path.join('data', 'output', 'loan_scores.pkl'))
    credit_score = probability_to_score_v3(predictions[0])

    # Mostrar resultado
    color = credit_score_range_classification(credit_score)
    st.markdown(
        f'<h2 style="color:{color};">Puntaje de Crédito Estimado: {credit_score:.2f}</h2>',
        unsafe_allow_html=True
    )

    # Generar gráfico
    buffer = plot_credit_score_distribution(y_scores, credit_score)
    image_data = buffer.getvalue()
    buffer.close()

    st.image(image_data, caption="Distribución de Puntajes de Crédito", use_container_width=True)
