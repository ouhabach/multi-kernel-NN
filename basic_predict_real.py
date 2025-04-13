import matplotlib.pyplot as plt
import numpy as np

# Calcul des valeurs réelles et prédites
real_values = y_test
predicted_values = y_pred

# Calcul des pertes
loss = np.abs(real_values - predicted_values) / np.maximum(real_values, predicted_values)
loss_abs = np.abs(real_values - predicted_values)

# Affichage des pertes moyennes
print(f"loss: {np.mean(loss)}")
print(f"loss abs: {np.mean(loss_abs)}")

# Tracer la courbe de régression
plt.figure(figsize=(8, 6))
plt.scatter(real_values, predicted_values, alpha=0.7, color='blue', label='Prédictions')
plt.plot([min(real_values), max(real_values)], [min(real_values), max(real_values)], color='red', linestyle='--', label='Régression idéale')
plt.title("Réel vs Prédit - Sum Rate Maximal")
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.legend()
plt.grid(True)

# Enregistrement du graphique en haute définition
plt.savefig("reel_vs_predit.png", dpi=300, bbox_inches='tight')

plt.show()

