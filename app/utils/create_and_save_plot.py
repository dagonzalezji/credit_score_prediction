import matplotlib.pyplot as plt
from io import BytesIO

def plot_credit_score_distribution(scores, point):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, color='green', alpha=0.6)
    plt.yscale('log')
    plt.axvline(x=point, color='red', linestyle='--', label=f'Puntaje {point}')
    plt.scatter(point, 1, color='red', s=100, zorder=5)
    plt.title("Distribución de Puntajes de Crédito (300-850)")
    plt.xlabel("Puntaje")
    plt.ylabel("Frecuencia (escala logarítmica)")
    plt.grid(True)
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)  # Mover el puntero al inicio del buffer
    plt.close()  # Cerrar la figura para liberar memoria

    return buffer