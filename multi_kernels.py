import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models,  backend as K
from tensorflow.keras.layers import Input, Lambda
import itertools
import tensorflow
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import numpy as np

import numpy as np
from itertools import combinations

def waterfilling_noma(channel_gains, selected_ris_channel, bs_to_ris_channel, total_power, RIS_users, noise_power=1e-9):
    """
    Allocation de puissance type "inverse water-filling" adaptée à NOMA downlink.
    On favorise les utilisateurs faibles pour respecter la logique NOMA.

    :param channel_gains: Liste des gains de canal (valeurs complexes) pour les utilisateurs.
    :param selected_ris_channel: Liste des canaux RIS pour les utilisateurs sélectionnés.
    :param bs_to_ris_channel: Canal de la BS vers chaque RIS.
    :param total_power: Puissance totale disponible.
    :param RIS_users: Nombre d'utilisateurs passant par le RIS.
    :return: Vecteur des puissances allouées.
    """
    # Puissances effectives des canaux |h|^2 (canal direct + composante RIS)
    channel_gains_power = np.abs(channel_gains) ** 2
    ris_gains_power = np.abs(selected_ris_channel) ** 2 * np.abs(bs_to_ris_channel[:RIS_users]) ** 2

    for ris in range(RIS_users):
        channel_gains_power[ris] += ris_gains_power[ris]

    # Inverse Water-Filling : plus le canal est faible, plus on lui alloue de puissance
    weights = 1 / (channel_gains_power + 1e-12)  # éviter division par 0
    weights /= np.sum(weights)  # normalisation

    power_allocation = weights * total_power
    return power_allocation


# Fonction de calcul de la capacité NOMA
def calculate_noma_capacity(channel_gains, selected_ris_channel, bs_to_ris_channel, power_allocation,RIS_users, noise_power=1e-9):
    """
    Calcule le débit total pour une combinaison de 3 utilisateurs en utilisant NOMA.
    L'utilisateur avec le canal le plus faible est décodé en premier.

    :param channel_gains: Liste des gains de canal (valeurs complexes) pour les utilisateurs.
    :param power_allocation: Liste des puissances allouées (doit être de taille 3).
    :param noise_power: Puissance du bruit (par défaut très faible).
    :return: Débits des utilisateurs (en bits/s/Hz).
    """
    #print(f"Dimensions de channel_gains: {len(channel_gains)}")
    #print(f"Dimensions de power_allocation: {len(power_allocation)}")
    #print(f"Dimensions de selected_ris_channel: {len(selected_ris_channel)}")
    #print(f"Dimensions de bs_to_ris_channel: {len(bs_to_ris_channel)}")
    # Calculer les puissances de canal (module au carré)
    channel_gains_power = np.abs(channel_gains) ** 2
    ris_gains_power = np.abs(selected_ris_channel) ** 2 * np.abs(bs_to_ris_channel) ** 2
    for ris in range(RIS_users):
        channel_gains_power[ris]+=ris_gains_power[ris]
    # Trier les indices par puissance de canal croissante (l'utilisateur avec le canal le plus faible en premier)
    sorted_indices = np.argsort(channel_gains_power)

    # Réorganiser les gains et les puissances en fonction des indices triés
    sorted_gains = channel_gains_power[sorted_indices]
    sorted_powers = [power_allocation[i] for i in range(len(sorted_indices))]

    capacities = np.zeros(len(channel_gains))

    for i, h_i in enumerate(sorted_gains):
        # Signal utile pour l'utilisateur i
        signal_power = sorted_powers[i] * h_i

        # Interférence (somme des puissances des utilisateurs avec index supérieur)
        interference = sum(sorted_powers[j] * sorted_gains[j] for j in range(i + 1, len(sorted_gains)))

        # SINR pour l'utilisateur i
        sinr = signal_power / (interference + noise_power)

        # Capacité en bits/s/Hz
        capacities[sorted_indices[i]] = np.log2(1 + sinr)

    return capacities


# Fonction pour optimiser l'allocation de puissance
def optimize_power_allocation(channel_gains, selected_ris_channel, bs_to_ris_channel, P_total,RIS_users, noise_power=1e-9):
    """
    Optimisation de l'allocation de puissance pour maximiser la capacité totale en NOMA.

    :param channel_gains: Liste des gains de canal pour les utilisateurs.
    :param P_total: Puissance totale disponible.
    :param noise_power: Puissance du bruit (par défaut très faible).
    :return: Allocation optimale de puissance.
    """
    # Fonction pour calculer la capacité totale pour une allocation donnée de puissances
    def total_capacity(power_allocation):
        capacities = calculate_noma_capacity(channel_gains, selected_ris_channel, bs_to_ris_channel, power_allocation,RIS_users, noise_power)
        return -np.sum(capacities)  # Minimiser la négative de la capacité totale

    # Initialiser les puissances allouées de manière égale
    initial_guess = np.ones(len(channel_gains)) * (P_total / len(channel_gains))

    # Contraintes : la somme des puissances doit être égale à la puissance totale
    constraints = ({'type': 'eq', 'fun': lambda power_allocation: np.sum(power_allocation) - P_total})

    # Bornes : les puissances doivent être positives
    bounds = [(0, P_total) for _ in channel_gains]

    # Utilisation de scipy.optimize.minimize pour résoudre l'optimisation
    result = minimize(total_capacity, initial_guess, bounds=bounds, constraints=constraints)

    # Résultat de l'optimisation : les puissances optimales allouées
    optimal_power_allocation = result.x

    return optimal_power_allocation






def sum_rate_from_binary_code(channels, ris_channel, bs_ris_channel, binary_code, n_users,nmax,  n_ris,n_ris_max, power_allocation=[0.4, 0.4, 0.4], noise_power=1e-9):
    """
    Calcule le sum rate des utilisateurs activés selon un code binaire avec au plus 6 utilisateurs actifs.
    Le sum rate maximal est retourné pour toutes les combinaisons de 3 utilisateurs parmi les utilisateurs actifs.

    :param channels: Liste des canaux complexes des utilisateurs (chaque canal est une valeur complexe).
    :param binary_code: Code binaire de N bits, où chaque bit spécifie si l'utilisateur est actif (1) ou non (0).
    :param power_allocation: Allocation de puissance pour les utilisateurs actifs (doit être de taille égale au nombre d'utilisateurs actifs).
    :param noise_power: Puissance du bruit (par défaut très faible).
    :return: Le sum rate maximal des utilisateurs actifs en bits/s/Hz.
    """
    binary_code_user = binary_code[:n_users]
    binary_code_ris = binary_code[n_users:]
    #bs_users,ris_to_users_channels,bs_to_ris_channel = load_complex_vectors(channels, selected_ris_channel, bs_to_ris_channel,n, n_ris)
    active_users_indices = [i for i, bit in enumerate(binary_code_user.decode('utf-8')) if bit == '1']
    active_ris_indices = [i for i, bit in enumerate(binary_code_ris.decode('utf-8')) if bit == '1']
    # Vérifier que le nombre d'utilisateurs actifs est entre 3 et 10
    if len(active_users_indices) < nmax or len(active_users_indices) > n_users:
        raise ValueError("Le code binaire doit activer entre 3 et 10 utilisateurs.")
    if len(active_ris_indices) < n_ris_max or len(active_ris_indices) > n_ris:
        raise ValueError("Le code binaire doit activer entre 3 et 10 utilisateurs.")

    # Liste pour stocker les sum rates des différentes combinaisons de 3 utilisateurs
    max_sum_rate = -float('inf')
    # Puissance totale disponible
    P_total = 10  # en Watts
    best_sum_rate = -np.inf
    # Tester toutes les combinaisons de 3 utilisateurs parmi les utilisateurs actifs
    for combination in combinations(active_users_indices, nmax):
       for ris_users in list(itertools.permutations(combination, n_ris_max)):
          ris_channels = np.zeros(n, dtype=complex)
          bs_to_ris_channels = np.zeros(n, dtype=complex)  # Canaux BS-à-RIS
          bs_to_user_channels = np.zeros(n, dtype=complex)  # Canaux BS-à-RIS

          for user in combination:
              bs_to_user_channels[user] = channels[user]  # Allouer les canaux BS-à-utilisateur

          ris_channels =  [ris_channel[i][idx] for idx,i in enumerate(ris_users)]
          bs_to_ris_channels =  [bs_ris_channel[idx] for idx,i in enumerate(ris_users)]
          '''
          for ris_idx in range(n_ris):  # Boucle sur tous les RIS
            for  i in ris_users:  # Boucle sur tous les utilisateurs
              #print(ris_idx,i)
              ris_channels[i] = ris_channel[i][ris_idx]  # Canal RIS à utilisateur
              bs_to_ris_channels[i] = bs_ris_channel[ris_idx]  # Canal BS-to-RIS pour l'utilisateur i
          '''
          # Optimiser l'allocation de puissance pour NOMA
          optimal_power_allocation = optimize_power_allocation(bs_to_user_channels,ris_channels,bs_to_ris_channels,P_total,n_ris_max)
          #optimal_power_allocation = waterfilling_noma(bs_to_user_channels, ris_channels, bs_to_user_channels, P_total, n_ris_max)
          #optimal_power_allocation = waterfilling_noma(bs_to_user_channels, ris_channels, bs_to_ris_channels, P_total, n_ris_max)

          # Calculer le taux total pour cette combinaison
          max_rates = calculate_noma_capacity(bs_to_user_channels,ris_channels,bs_to_ris_channels, optimal_power_allocation,n_ris_max)
          total_sum_rate = np.sum(max_rates)
          if total_sum_rate > best_sum_rate:
            best_sum_rate = total_sum_rate

    return best_sum_rate



def load_complex_vectors(channels, n, RIS_users):
    """
    Lit le fichier CSV et extrait les vecteurs complexes pour CSI User, RIS to User et BS to RIS.

    Args:
    - filename: Le chemin vers le fichier CSV.
    - N: Le nombre d'utilisateurs.
    - RIS_users: Le nombre de RIS.

    Returns:
    - csi_users: Un tableau numpy des vecteurs complexes pour chaque utilisateur.
    - ris_to_users: Un tableau numpy des vecteurs complexes pour chaque RIS-to-User.
    - bs_to_ris: Un tableau numpy des vecteurs complexes pour chaque BS-to-RIS.
    """


    # Initialiser les listes pour les vecteurs complexes
    csi_users = []
    ris_users = []
    bs_to_ris = []

    # Extraire les colonnes complexes pour CSI User (CSI_User_{i+1}_Re et CSI_User_{i+1}_Im)
    for i in range(n):
        real_part = channels[2+3*i][0]
        imag_part = channels[2+3*i][1]
        csi_users.append(real_part + 1j * imag_part)  # Créer un vecteur complexe pour chaque utilisateur
    #print(n,csi_users)
    # Extraire les colonnes complexes pour RIS to User (RIS_to_User_{i+1}_RIS{ris_idx+1}_Re et _Im)
    for i in range(n):
        ris_to_users = []
        for ris_idx in range(2):
            real_part = channels[3+3*i+ris_idx][0]
            imag_part = channels[3+3*i+ris_idx][1]
            ris_to_users.append(real_part + 1j * imag_part)  # Créer un vecteur complexe pour chaque RIS to User
        ris_users.append(ris_to_users)

    # Extraire les colonnes complexes pour BS to RIS (BS_to_RIS_{ris+1}_Re et _Im)
    for ris in range(2):
        real_part = channels[ris][0]
        imag_part = channels[ris][1]
        bs_to_ris.append(real_part + 1j * imag_part)  # Créer un vecteur complexe pour chaque BS to RIS

    # Convertir les listes en tableaux numpy
    csi_users = np.array(csi_users)  # Dimension: (N, nombre_total_d'échantillons)
    ris_users_ = np.array(ris_users).reshape(-1, 2)  # Dimension: (N * RIS_users, nombre_total_d'échantillons)
    bs_to_ris = np.array(bs_to_ris)  # Dimension: (RIS_users, nombre_total_d'échantillons)

    return csi_users, ris_users_, bs_to_ris





def decimal_maximal(n):
    # Le nombre maximal en binaire à n bits est un nombre composé uniquement de 1
    # Cela correspond à (2^n) - 1
    return (2 ** n) - 1


def binaire_to_decimal(binaire, n_bits):
    if len(binaire) != n_bits:
        raise ValueError(f"Le code binaire doit être de {n_bits} bits")
    return int(binaire, 2)

def codes_binaires_avec_n_1(k, n):
    if n > k:
        raise ValueError("Le nombre de 1 ne peut pas être plus grand que le nombre total de bits.")

    # Trouver toutes les combinaisons d'indices où les "1" doivent être placés
    indices_1 = itertools.combinations(range(k), n)

    # Générer les codes binaires à partir de ces indices
    codes = []
    for indices in indices_1:
        code = ['0'] * k  # Code binaire initial (tout est à 0)
        for index in indices:
            code[index] = '1'  # Placer un "1" aux indices spécifiés
        codes.append(''.join(code))  # Joindre la liste en une chaîne de caractères

    return codes

def extraire_canaux(vecteur_canaux, code_binaire, ris_binaire, n_ris):
    # Vérification que le code binaire et le vecteur de canaux ont la même longueur
    if len(vecteur_canaux) != n_ris + (n_ris+1)*len(code_binaire):
        raise ValueError("La longueur du code binaire doit être égale à la longueur du vecteur de canaux")

    #print(f"code_binaire: {len(code_binaire)}")
    # Liste pour stocker les canaux extraits
    canaux_extraits = []
    for j, rbit in enumerate(ris_binaire):
      if rbit == '1':  # Si le bit est 1, on extrait le canal correspondant
          canaux_extraits.append(vecteur_canaux[j])
    #print(f"Dimensions de canaux_extraits: {len(canaux_extraits)}
    # Parcourir chaque canal et le code binaire associé
    for i, bit in enumerate(code_binaire):
      if bit == '1':  # Si le bit est 1, on extrait le canal correspondant
              canaux_extraits.append(vecteur_canaux[n_ris+(n_ris+1)*i])
              for j, rbit in enumerate(ris_binaire):
                  if rbit == '1':  # Si le bit est 1, on extrait le canal correspondant
                      canaux_extraits.append(vecteur_canaux[n_ris+(n_ris+1)*i+j])


    #print(f"Dimensions de canaux_extraits: {len(canaux_extraits)}")
    return canaux_extraits


# Charger le modèle pré-entraîné pour 3 utilisateurs
def load_pretrained_model(weights_path, input_dim):
    model = models.Sequential([
        #layers.Dense(128, input_dim=input_dim, activation='relu'),
        layers.Dense(64, input_dim=input_dim, activation='relu'),
        #layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Sortie avec 3 neurones (un pour chaque utilisateur)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.load_weights(weights_path)  # Charger les poids du modèle
    return model

# Créer un modèle 3D pour n utilisateurs
def create_n_user_model(n, n_ris, input_dim, pretrained_weights_path,nmax,n_ris_max):
    inputs = [layers.Input(shape=(2,)) for _ in range((n_ris+1)*n+n_ris)]  # Entrées pour chaque utilisateur
    # Charger le modèle pré-entraîné (qui prend 3 utilisateurs et génère 3 sorties)
    base_model = load_pretrained_model(pretrained_weights_path, input_dim)
    nris_model = 2
    # Créer toutes les combinaisons de 3 utilisateurs parmi les n utilisateurs
    #user_combinations = generate_lots(n,6,3)
    #ris_combinations = generate_lots(n_ris,2,2)

    user_combinations = generate_lots(n,nmax,nmax)
    ris_combinations = generate_lots(n_ris,n_ris_max,n_ris_max)
    # Liste pour stocker les sorties
    final_outputs = []
    combinations = []
    for comb in user_combinations:
      for ris in ris_combinations:
          # Obtenez les entrées pour les utilisateurs dans cette combinaison
          user_inputs = extraire_canaux(inputs, comb,ris, n_ris)
          # Concaténer les entrées des 3 utilisateurs pour cette combinaison (forme (None, 6))
          concatenated_input = layers.Concatenate()(user_inputs)
          # Appliquer le modèle pré-entraîné sur l'entrée concaténée
          outputs = base_model(concatenated_input)  # Sorties pour cette combinaison de 3 utilisateurs
          # Ajouter la sortie sommée à final_outputs
          final_outputs.append(outputs)
          combinations.append(comb+ris)  # Stocker la combinaison correspondante

    # Empiler les sorties pour passer à la réduction (max)
    stacked_outputs = layers.Concatenate()(final_outputs)  # Forme (None, num_combinations)

    # Calculer le max sur l'axe des combinaisons (axis=-1)
    max_outputs = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1),output_shape=(1,))(stacked_outputs)


    # Appliquer argmax pour obtenir l'indice du maximum
    max_indices = layers.Lambda(lambda x: tf.argmax(x, axis=-1),output_shape=(1,))(stacked_outputs)


    # Obtenir l'indice de la combinaison maximale
    # Définir la forme de sortie du gather pour qu'elle soit compatible avec la structure
    max_combination_index = layers.Lambda(lambda x: tf.gather(combinations, x),output_shape=(3,))(max_indices)

    # Créer le modèle final avec les entrées et les sorties
    final_model = models.Model(inputs=inputs, outputs=[max_outputs, max_combination_index])

    return final_model



# Préparer les entrées correctement pour chaque utilisateur (pour 4 utilisateurs)
def prepare_user_inputs(X,n,n_ris):
    # Diviser X en 4 sous-entrées pour chaque utilisateur (2 éléments CSI par utilisateur)
    user_inputs = [X[:, i:i+2] for i in range(0, X.shape[1], 2)]  # Diviser en 4 utilisateurs (2 éléments par utilisateur)

    # Maintenant nous allons concaténer les 3 utilisateurs pour chaque combinaison
    inputs_3_users = []
    user_combinations = list(itertools.combinations(range(n), 3))  # Combinaison de 3 utilisateurs parmi les 4
    for comb in user_combinations:
        # Rassembler les 2 éléments CSI pour chaque utilisateur dans la combinaison
        concatenated_input = np.concatenate([user_inputs[i] for i in comb], axis=1)  # (None, 6)
        inputs_3_users.append(concatenated_input)

    # Convertir en tableau numpy
    return user_inputs#np.concatenate(inputs_3_users, axis=0)  # Combinaison des entrées des utilisateurs (forme: (None, 6))


# Faire des prédictions sur le testset
def make_predictions_on_testset(model, X_test,n,n_ris):
    # Préparer les entrées correctement pour les utilisateurs
    user_inputs = prepare_user_inputs(X_test,n,n_ris)
    # Faire des prédictions
    y_pred = model.predict(user_inputs)

    return y_pred






# Fonction pour charger les données depuis un fichier CSV
def load_data(filename, n_users, n_ris):
    """
    Charge les données depuis un fichier CSV pour n utilisateurs et n RIS, et sépare les caractéristiques (features)
    et les cibles (targets).
    """
    df = pd.read_csv(filename)

    # Dynamique: sélection des colonnes pour BS to RIS
    csi_columns = []
    RIS_bs = []
    RIS_users = []
     # Générer les colonnes pour chaque utilisateur (chaque canal)
    for i in range(1, n_ris + 1):
        RIS_bs.append(f"BS RIS{i}_real")
        RIS_bs.append(f"BS RIS{i}_imag")

    for i in range(1, n_users + 1):
        csi_columns.append(f"CSI user{i}_real")
        csi_columns.append(f"CSI user{i}_imag")

    for ris in range(1, n_ris + 1):
        for i in range(1, n_users + 1):
            RIS_users.append(f"RIS user{ris}_{i}_real")
            RIS_users.append(f"RIS user{ris}_{i}_imag")

    # Extraire les caractéristiques X
    X = df[csi_columns].values
    X_bs = df[RIS_bs].values
    X_ris = df[RIS_users].values
    # La cible (sortie) est la somme des sum-rates
    y = df["sum-rate max"].values

    return X,X_bs,X_ris, y

def load_X_cplx(X,n,n_ris):
  X_cplx = []
  for u in range(n):
    X_cplx.append(complex(X[u*2],X[u*2+1]))
  return X_cplx

def load_X_bs_cplx(X,n,n_ris):
  X_cplx = []
  for i in range(n_ris):
    X_cplx.append(complex(X[i*2],X[i*2+1]))
  return X_cplx

def load_X_ris_cplx(X, n, n_ris):
    X_cplx = [[] for _ in range(n)]  # Initialize empty lists for each RIS
    for u in range(n):
        # Print index information
        for i in range(n_ris):
            # Calculate the index for real and imaginary parts
            real_idx = (i * n + u) * 2  # Real part at even indices
            imag_idx = real_idx + 1     # Imaginary part at odd indices

            # Check if indices are within bounds
            if real_idx < len(X) and imag_idx < len(X):
                # Append the complex number to the corresponding RIS list
                X_cplx[u].append(complex(X[real_idx], X[imag_idx]))
            else:
                print(f"Warning: Index out of bounds for user {u} and RIS {i}")

    return X_cplx

# Fonction pour générer tous les triplets parmi n utilisateurs
def generate_triplets(n,l):
    return list(itertools.combinations(range(1, n + 1), l))

# Fonction pour générer les lots de taille 6 en format binaire
def generate_lots(n, k=6, l=3):
    # Générer tous les triplets possibles parmi les n utilisateurs
    triplets = generate_triplets(n,l)

    # Initialisation de la liste des lots
    lots = []

    # Suivi des triplets couverts
    covered_triplets = set()

    # Ajouter des lots jusqu'à ce que tous les triplets soient couverts
    while triplets:
        # Initialiser un lot vide
        lot = set()

        # Ajouter des triplets dans le lot jusqu'à ce qu'il ait exactement 6 utilisateurs
        for triplet in triplets[:]:
            # Si le lot peut inclure ce triplet sans dépasser 6 utilisateurs
            if len(lot.union(triplet)) <= k:
                lot.update(triplet)
                triplets.remove(triplet)

        # Si le lot a exactement 6 utilisateurs, l'ajouter à la liste des lots
        if len(lot) == k:
            # Convertir le lot en une chaîne binaire de taille n
            binary_lot = ['0'] * n  # Initialiser une liste de n zéros
            for user in lot:
                binary_lot[user - 1] = '1'  # Mettre 1 pour chaque utilisateur dans le lot

            # Ajouter la chaîne binaire du lot à la liste des lots
            lots.append(''.join(binary_lot))

    return lots


n = 7  # Nombre d'utilisateurs (15 utilisateurs dans ce cas)
n_ris=3
nmax= 5
n_ris_max = 3

test_size = 1000
input_dim = 2*(n_ris_max+nmax*(1+n_ris_max))  # Chaque vecteur d'entrée a 6 éléments (2 éléments CSI pour chaque utilisateur dans une combinaison de 3)
weights_path = 'model.weights.h5'  # Chemin vers les poids enregistrés du modèle pré-entraîné
power_allocation = [0.4, 0.4, 0.4]  # Allocation de puissance
# Charger le modèle 3D avec les poids pré-entraînés
model_3d = create_n_user_model(n,n_ris, input_dim, weights_path,nmax,n_ris_max)

# Charger les données de test depuis le fichier CSV
testset_file = 'train53set7-3-1000.csv'  # Remplacer par le chemin de votre fichier CSV

# Charger les données
X,X_bs,X_ris, y = load_data(testset_file,n,n_ris)

print(f"Dimensions de X: {X.shape}")
print(f"Dimensions de y: {y.shape}")
scaler_y = StandardScaler()
y_test_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

# Appliquer la standardisation (moyenne = 0, écart type = 1)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X)  # Normalisation des entrées

# Appliquer la standardisation (moyenne = 0, écart type = 1)
scaler_bs = StandardScaler()
X_bs_scaled = scaler_bs.fit_transform(X_bs)  # Normalisation des entrées

# Appliquer la standardisation (moyenne = 0, écart type = 1)
scaler_ris = StandardScaler()
X_ris_scaled = scaler_ris.fit_transform(X_ris)  # Normalisation des entrées
X_test_scaledX = np.concatenate((X_test_scaled, X_bs_scaled,X_ris_scaled), axis=1)
# Faire des prédictions sur le testset
y_pred_scaled = make_predictions_on_testset(model_3d, X_test_scaledX,n,n_ris)
# Extraire les sum_rates (premier élément de y_pred_scaled)
sum_rate_scaled = y_pred_scaled[0]

# Extraire les codes binaires (deuxième élément de y_pred_scaled)
code_binaries_scaled = y_pred_scaled[1]
# Conversion en tableau 2D (nécessaire pour le scaler)
sum_rate_values_2d = np.array(sum_rate_scaled).reshape(-1, 1)  # .reshape(-1, 1) pour convertir en 2D
y_pred_sum_rate = scaler_y.inverse_transform(sum_rate_values_2d)
# Sauvegarder les poids dans un fichier .h5
model_3d.save_weights("model3D30.weights.h5")
# Sauvegarder le modèle complet
model_3d.save("modele3D30_complet.h5")
loss = 0
for i in range(5):  # Afficher les premières prédictions et valeurs réelles
  X_tst = load_X_cplx(X[i,:],n,n_ris)
  X_bs_tst = load_X_bs_cplx(X_bs[i,:],n,n_ris)
  X_ris_tst = load_X_ris_cplx(X_ris[i,:],n,n_ris)
  real_value = y[i]  # La valeur réelle (premier élément de y_test2[i])
  predicted_value = sum_rate_from_binary_code(X_tst,X_ris_tst,X_bs_tst, code_binaries_scaled[i],n,nmax,n_ris,n_ris_max, power_allocation)

  # Calculer la perte (erreur absolue)
  loss += abs(real_value - predicted_value)/max(real_value , predicted_value)

  # Afficher la valeur réelle, la prédiction et la perte
  print(f"Valeur réelle: {real_value} - Valeur prédite: {code_binaries_scaled[i]} ")
  print(f"Réel: {real_value} - valeur réelle du prédit {predicted_value,code_binaries_scaled[i]}")
