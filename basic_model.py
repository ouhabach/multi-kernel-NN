import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import joblib  # Pour sauvegarder le scaler



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

    # Extraire les caractéristiques RIS
    RIS = df[csi_columns].values

    csi_columns = []
    ris_columns = []
    # Ajouter les colonnes CSI pour chaque utilisateur
    for user in range(1, n_users + 1):
        # Canaux CSI du BS vers l'utilisateur
        csi_columns.append(f"CSI user{user}_real")
        csi_columns.append(f"CSI user{user}_imag")

        # Canaux entre RIS et l'utilisateur (RIS1 à RISn_ris pour chaque utilisateur)
        for ris in range(1, n_ris + 1):
            ris_columns.append(f"RIS user{ris}_{user}_real")
            ris_columns.append(f"RIS user{ris}_{user}_imag")

    # Extraire les caractéristiques X
    X = df[csi_columns].values
    Xris = df[ris_columns].values



    # La cible (sortie) est la somme des sum-rates
    y = df[["sum-rate max"]].values

    return RIS, X,Xris, y



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
        layers.Dense(64, input_dim=input_dim),layers.LeakyReLU(alpha=0.01),
        layers.Dense(128),layers.LeakyReLU(alpha=0.01),
        layers.Dense(64),layers.LeakyReLU(alpha=0.01),
        layers.Dense(32),layers.LeakyReLU(alpha=0.01),
        layers.Dense(16),layers.LeakyReLU(alpha=0.01),
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

RIS, X,Xris, y = load_data_from_csv(filename,n,n_ris)

print(f"Dimensions de X: {X.shape}")
print(f"Dimensions de X: {Xris.shape}")
print(f"Dimensions de y: {y.shape}")
# Diviser les données en ensemble d'entraînement (70%) et de test (30%)
RIS_train, RIS_test, X_train, X_test,Xris_train, Xris_test, y_train, y_test = train_test_split(RIS, X,Xris, y, test_size=0.3, random_state=42)

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

# Appliquer la standardisation (moyenne = 0, écart type = 1)
scalerRuser = StandardScaler()
Xris_train_scaled = scalerRuser.fit_transform(Xris_train)  # Normalisation des entrées
Xris_test_scaled = scalerRuser.transform(Xris_test)

# Appliquer la standardisation (moyenne = 0, écart type = 1)
scalerRIS = StandardScaler()
RIS_train_scaled = scalerRIS.fit_transform(RIS_train)  # Normalisation des entrées
RIS_test_scaled = scalerRIS.transform(RIS_test)

XT_scaled = np.hstack((RIS_train_scaled,X_train_scaled,Xris_train_scaled))
XS_scaled = np.hstack((RIS_test_scaled,X_test_scaled,Xris_test_scaled )) 
# Créer le modèle de réseau de neurones
model = create_nn_model(XT_scaled.shape[1])

# Entraîner le modèle
model.fit(XT_scaled, y_train_scaled, epochs=80, batch_size=32, validation_data=(XS_scaled, y_test_scaled))

# Évaluer le modèle sur l'ensemble de test
test_loss = model.evaluate(XS_scaled, y_test_scaled)
print(f"Test Loss: {test_loss}")

# Prédire sur l'ensemble de test
y_pred_scaled = model.predict(XS_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)


# Afficher les prédictions et les valeurs réelles
for i in range(5):  # Afficher les premières prédictions et valeurs réelles
    print(f"Réel: {y_test[i]} - Prédit: {y_pred[i]}")

# Sauvegarder les poids dans un fichier .h5
model.save_weights("model.weights.h5")

# Sauvegarder le modèle complet
model.save("mon_modele_complet.h5")

# Sauvegarder le scaler
joblib.dump(scaler_y, 'scaler_y.pkl')

# Sauvegarder le scaler
joblib.dump(scaler, 'scaler_x.pkl')

# Sauvegarder le scaler
joblib.dump(scalerRIS, 'scaler_ris.pkl')

# Sauvegarder le scaler
joblib.dump(scalerRuser, 'scaler_ruser.pkl')
