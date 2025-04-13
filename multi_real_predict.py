import matplotlib.pyplot as plt
import numpy as np

# Calcul des valeurs réelles et prédites
real_values = y
predicted_values = np.array([
    sum_rate_from_binary_code(
        load_X_cplx(X[i, :], n, n_ris),
        load_X_ris_cplx(X_ris[i, :], n, n_ris),
        load_X_bs_cplx(X_bs[i, :], n, n_ris),
        code_binaries_scaled[i], n,nmax,n_ris,n_ris_max, power_allocation
    ) for i in range(len(y))
])

# Calcul des pertes
loss = np.abs(real_values - predicted_values) / np.maximum(real_values, predicted_values)
loss_abs = np.abs(real_values - predicted_values)

loss_abs_sum = np.sum(loss_abs)
# Affichage des pertes moyennes
print(f"loss: {np.mean(loss)}")
print(f"loss abs: {np.mean(loss_abs)}")
print(f"loss_abs_sum: {np.mean(loss_abs_sum)}")
# Tracer la courbe de régression
plt.figure(figsize=(8, 6))
plt.scatter(real_values, predicted_values, alpha=0.7, color='blue', label='Prédictions')
plt.plot([min(real_values), max(real_values)], [min(real_values), max(real_values)], color='red', linestyle='--', label='Régression idéale')
plt.title("Réel vs Prédit - Sum Rate Maximal")
plt.xlabel("Valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.legend()
plt.grid(True)
plt.show()
