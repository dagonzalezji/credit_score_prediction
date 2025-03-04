import numpy as np


def probability_to_score_v3(prob, base_score=300, max_score=850, threshold=0.326,
                                   expansion_factor_low=3, expansion_factor_high=0.7):
    """
    Convierte probabilidades en puntajes de crédito con expansión no lineal
    para distribuir mejor en los extremos.

    Args:
        prob (float): Probabilidad de default.
        base_score (int): Puntaje base.
        max_score (int): Puntaje máximo.
        threshold (float): Valor de corte óptimo.
        expansion_factor_low (float): Factor para expandir la parte baja del rango.
        expansion_factor_high (float): Factor para expandir la parte alta del rango.

    Returns:
        score (float): Puntaje de crédito ajustado.
    """
    # Invertir la probabilidad para que mayor valor sea mejor score
    inverted_prob = 1 - prob

    # Punto de corte invertido
    inverted_threshold = 1 - threshold

    # Determinar si es un score alto o bajo
    if inverted_prob >= inverted_threshold:  # Buenos clientes
        # Normalizar la probabilidad en el rango de buenos
        normalized = (inverted_prob - inverted_threshold) / (1 - inverted_threshold)
        # Aplicar expansión no lineal
        transformed = normalized ** expansion_factor_high
        # Mapear al rango superior
        mid_score = 550  # Punto medio del rango
        score = mid_score + (max_score - mid_score) * transformed
    else:  # Malos clientes
        # Normalizar la probabilidad en el rango de malos
        normalized = inverted_prob / inverted_threshold
        # Aplicar expansión no lineal para los scores bajos
        transformed = normalized ** expansion_factor_low
        # Mapear al rango inferior
        mid_score = 550  # Punto medio del rango
        score = base_score + (mid_score - base_score) * transformed

    # Asegurar que el score esté dentro del rango permitido
    score = np.clip(score, base_score, max_score)

    return score