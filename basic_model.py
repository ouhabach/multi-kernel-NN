import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from itertools import combinations
from sklearn.preprocessing import StandardScaler



import pandas as pd

def load_data_from_csv(filename, n_users, n_ris):
    """
    Charge les données depuis un fichier CSV pour n utilisateurs et n RIS, et sépare les caractéristiques (features)
    et les cibles (targets).

    Args:
    - filename: Chemin du fichier CSV.
    - n_users: Le nombre d'utilisateurs.
    - n_ris: Le nombre de RIS dans le système.

    Returns:
    - X: Les caractéristiques (entrées) du modèle.
    - y: Les cibles (sorties) du modèle.
    """
    # Charger le fichier CSV dans un DataFrame
    df = pd.read_csv(filename)

    # Dynamique: sélection des colonnes pour BS to RIS
    csi_columns = []
    for ris in range(1, n_ris + 1):  # BS to RIS1, BS to RIS2, ...
        csi_columns.append(f"BS RIS{ris}_real")
        csi_columns.append(f"BS RIS{ris}_imag")

    # Ajouter les colonnes CSI pour chaque utilisateur
    for user in range(1, n_users + 1):
        # Canaux CSI du BS vers l'utilisateur
        csi_columns.append(f"CSI user{user}_real")
        csi_columns.append(f"CSI user{user}_imag")

        # Canaux entre RIS et l'utilisateur (RIS1 à RISn_ris pour chaque utilisateur)
        for ris in range(1, n_ris + 1):
            csi_columns.append(f"RIS user{ris}_{user}_real")
            csi_columns.append(f"RIS user{ris}_{user}_imag")

    # Extraire les caractéristiques X
    X = df[csi_columns].values

    # La cible (sortie) est la somme des sum-rates
    y = df[["sum-rate max"]].values

    return X, y



# Fonction pour créer un modèle de réseau de neurones
def create_nn_model(input_dim):
    """
    Crée un modèle de réseau de neurones simple pour prédire la somme des sum-rates.

    Args:
    - input_dim: La dimension des caractéristiques d'entrée (features).

    Returns:
    - model: Un modèle Keras pour entraîner et prédire la somme des sum-rates.
    """
    model = models.Sequential([
        #layers.Dense(128, input_dim=input_dim, activation='relu'),
        layers.Dense(64, input_dim=input_dim, activation='relu'),
        #layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Sortie avec un seul neurone représentant la somme des sum-rates
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Fonction principale
#def main():

n= 5
n_ris = 3
# Charger les données depuis le fichier CSV
filename = "train53set5-100k.csv"  # Changez le nom de votre fichier CSV ici

X, y = load_data_from_csv(filename,n,n_ris)

print(f"Dimensions de X: {X.shape}")
print(f"Dimensions de y: {y.shape}")
# Diviser les données en ensemble d'entraînement (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Dimensions de X_train: {X_train.shape}")
print(f"Dimensions de X_test: {X_test.shape}")
print(f"Dimensions de y_train: {y_train.shape}")
print(f"Dimensions de y_test: {y_test.shape}")

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Appliquer la standardisation (moyenne = 0, écart type = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Normalisation des entrées
X_test_scaled = scaler.transform(X_test)

# Créer le modèle de réseau de neurones
model = create_nn_model(X_train.shape[1])

# Entraîner le modèle
model.fit(X_train_scaled, y_train_scaled, epochs=80, batch_size=32, validation_data=(X_test_scaled, y_test_scaled))

# Évaluer le modèle sur l'ensemble de test
test_loss = model.evaluate(X_test_scaled, y_test_scaled)
print(f"Test Loss: {test_loss}")

# Prédire sur l'ensemble de test
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)


# Afficher les prédictions et les valeurs réelles
for i in range(5):  # Afficher les premières prédictions et valeurs réelles
    print(f"Réel: {y_test[i]} - Prédit: {y_pred[i]}")

# Sauvegarder les poids dans un fichier .h5
model.save_weights("model.weights.h5")

# Sauvegarder le modèle complet
model.save("mon_modele_complet.h5")
