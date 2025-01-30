import numpy as np


def probability_to_score_v3(prob, base_score=300, max_score=850, pdo=50, midpoint=0.025, scale=0.5):
    """
    Convierte probabilidades en puntajes de crédito con ajustes para reducir la concentración en los extremos.

    Args:
        prob (float): Probabilidad de default.
        base_score (int): Puntaje base (ej. 300).
        max_score (int): Puntaje máximo (ej. 850).
        pdo (int): Puntos para doblar las probabilidades.
        midpoint (float): Punto de inflexión para la transformación logística.
        scale (float): Escala de la transformación logística.

    Returns:
        score (float): Puntaje de crédito ajustado al rango.
    """
    # Clipping para evitar problemas numéricos
    prob = np.clip(prob, 1e-6, 1 - 1e-6)

    # Transformación logística para una distribución más uniforme
    transformed_prob = 1 / (1 + np.exp(-(np.log(prob / (1 - prob)) - np.log(midpoint / (1 - midpoint))) / scale))

    # Calcular odds con la probabilidad transformada
    odds = (1 - transformed_prob) / transformed_prob

    # Calcular factor y offset
    factor = pdo / np.log(2)
    offset = base_score + (max_score - base_score) / 2  # Centrar el rango

    # Calcular puntaje crudo
    raw_score = offset - factor * np.log(odds)

    # Normalizar el puntaje dentro del rango [base_score, max_score]
    score = np.clip(raw_score, base_score, max_score)

    return score