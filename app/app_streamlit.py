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
preprocessor = joblib.load("app/models/preprocessor_2.pkl")

def probability_to_score(prob, base_score=300, max_score=850, pdo=50):
    odds = (1 - prob) / prob
    factor = pdo / np.log(2)
    offset = base_score + 200
    raw_score = offset - factor * np.log(odds)
    return np.clip(raw_score, base_score, max_score)

# Interfaz Streamlit
st.title("Predicción de Puntaje de Crédito")

st.sidebar.header("Ingrese los valores del préstamo")
recoveries = st.sidebar.number_input("Recoveries", value=0.0)
collection_recovery_fee = st.sidebar.number_input("Collection Recovery Fee", value=0.0)
total_rec_prncp = st.sidebar.number_input("Total Rec Prncp", value=0.0)
out_prncp = st.sidebar.number_input("Out Prncp", value=0.0)
last_pymnt_amnt = st.sidebar.number_input("Last Payment Amount", value=0.0)
total_pymnt = st.sidebar.number_input("Total Payment", value=0.0)
installment = st.sidebar.number_input("Installment", value=0.0)
funded_amnt_inv = st.sidebar.number_input("Funded Amount Inv", value=0.0)
loan_amnt = st.sidebar.number_input("Loan Amount", value=0.0)
total_rec_int = st.sidebar.number_input("Total Rec Int", value=0.0)
total_rec_late_fee = st.sidebar.number_input("Total Rec Late Fee", value=0.0)
int_rate = st.sidebar.number_input("Interest Rate", value=0.0)
inq_last_6mths = st.sidebar.number_input("Inquiries Last 6 Months", value=0)
open_acc = st.sidebar.number_input("Open Accounts", value=0)

# Opciones de selección
term = st.sidebar.selectbox("Term", ["36 months", "60 months"])
emp_length = st.sidebar.selectbox("Employment Length", ["< 1 year", "1-5 years", "6-10 years", "10+ years"])
home_ownership = st.sidebar.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT"])
purpose = st.sidebar.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement"])
grade = st.sidebar.selectbox("Grade", ["A", "B", "C", "D", "E", "F", "G"])
initial_list_status = st.sidebar.selectbox("Initial List Status", ["w", "f"])

# Tamaño de la muestra
#n_samples = 1000

# Generar datos numéricos
# numeric_data = {
#     "recoveries": np.random.uniform(0, 5000, n_samples),
#     "collection_recovery_fee": np.random.uniform(0, 1000, n_samples),
#     "total_rec_prncp": np.random.uniform(1000, 35000, n_samples),
#     "out_prncp": np.random.uniform(0, 35000, n_samples),
#     "last_pymnt_amnt": np.random.uniform(0, 1000, n_samples),
#     "total_pymnt": np.random.uniform(1000, 40000, n_samples),
#     "installment": np.random.uniform(100, 1500, n_samples),
#     "funded_amnt_inv": np.random.uniform(1000, 35000, n_samples),
#     "loan_amnt": np.random.uniform(1000, 35000, n_samples),
#     "total_rec_int": np.random.uniform(0, 8000, n_samples),
#     "total_rec_late_fee": np.random.uniform(0, 100, n_samples),
#     "int_rate": np.random.uniform(5, 25, n_samples),
#     "inq_last_6mths": np.random.randint(0, 10, n_samples),
#     "open_acc": np.random.randint(1, 30, n_samples),
# }
#
# # Generar datos categóricos
# categorical_data = {
#     "term": np.random.choice(["36 months", "60 months"], n_samples),
#     "emp_length": np.random.choice(
#         [
#             "< 1 year",
#             "1 year",
#             "2 years",
#             "3 years",
#             "4 years",
#             "5 years",
#             "6 years",
#             "7 years",
#             "8 years",
#             "9 years",
#             "10+ years",
#         ],
#         n_samples,
#     ),
#     "home_ownership": np.random.choice(
#         ["RENT", "OWN", "MORTGAGE", "OTHER"], n_samples
#     ),
#     "purpose": np.random.choice(
#         [
#             "debt_consolidation",
#             "credit_card",
#             "home_improvement",
#             "small_business",
#             "major_purchase",
#             "other",
#         ],
#         n_samples,
#     ),
#     "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples),
#     "initial_list_status": np.random.choice(["w", "f"], n_samples),
# }
#
# # Crear DataFrame
# df = pd.DataFrame({**numeric_data, **categorical_data})

if st.sidebar.button("Predecir"):
    #Preparar datos
    input_array = np.array([[recoveries, collection_recovery_fee, total_rec_prncp, out_prncp, last_pymnt_amnt,
                              total_pymnt, installment, funded_amnt_inv, loan_amnt, total_rec_int,
                              total_rec_late_fee, int_rate, inq_last_6mths, open_acc, term, emp_length,
                              home_ownership, purpose, grade, initial_list_status]])

    columns = ["recoveries", "collection_recovery_fee", "total_rec_prncp", "out_prncp", "last_pymnt_amnt",
               "total_pymnt", "installment", "funded_amnt_inv", "loan_amnt", "total_rec_int",
               "total_rec_late_fee", "int_rate", "inq_last_6mths", "open_acc", "term", "emp_length",
               "home_ownership", "purpose", "grade", "initial_list_status"]

    df = pd.DataFrame(input_array, columns=columns)

    # Preprocesar y predecir
    data_processed = preprocessor.transform(df)
    print(df)
    predictions = model.predict(data_processed).ravel()
    print(predictions)
    #y_pred_proba = (predictions > 0.5).astype(int)
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