import matplotlib.pyplot as plt
import numpy as np

# Calcul des valeurs réelles et prédites
real_values = y_test
predicted_values = y_pred

# Calcul des pertes
loss = np.abs(np.array(real_values) - np.array(predicted_values)) / np.maximum(np.array(real_values), np.array(predicted_values))
loss_abs = np.abs(real_values - predicted_values)

# Affichage des pertes moyennes
print(f"loss: {np.mean(loss)}")
print(np.array(real_values) - np.array(predicted_values))
print(np.argmax(loss))
print(real_values[18243]) 
print(predicted_values[18243]) 
print(f"loss: {np.max(loss)}")
print(f"loss abs: {np.mean(loss_abs)}")

# Tracer la courbe de régression
plt.figure(figsize=(8, 6))
plt.scatter(real_values, predicted_values, alpha=0.7, color='blue', label='Predicted')
plt.plot([min(real_values), max(real_values)], [min(real_values), max(real_values)], color='red', linestyle='--', label='Ideal regression')
#plt.title("Réel vs Prédit - Sum Rate Maximal")
plt.xlabel("Real Sum-rate")
plt.ylabel("Predicted Sum-rate")
plt.legend()
plt.grid(True)

# Enregistrement du graphique en haute définition
plt.savefig("reel_vs_predit.png", dpi=300, bbox_inches='tight')

plt.show()
