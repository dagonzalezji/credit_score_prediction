import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import base64
from io import BytesIO
from utils.create_and_save_plot import plot_credit_score_distribution

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

if st.sidebar.button("Predecir"):
    # Preparar datos
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
    predictions = model.predict(data_processed).ravel()
    y_pred_proba = (predictions > 0.5).astype(int)
    credit_score = probability_to_score(y_pred_proba[0])

    # Mostrar resultado
    st.subheader(f"Puntaje de Crédito Estimado: {credit_score:.2f}")

    # Generar gráfico
    buffer = plot_credit_score_distribution([credit_score], credit_score)
    image_data = buffer.getvalue()
    buffer.close()

    st.image(image_data, caption="Distribución de Puntajes de Crédito", use_column_width=True)