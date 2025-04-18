import itertools
import numpy as np
import csv
from itertools import combinations
from scipy.optimize import minimize


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




def noma_allocation(channels,bs_to_ris_channel,ris_to_users_channels, N,RIS_users, N_users):
    """
    Teste l'allocation des utilisateurs en utilisant NOMA.

    :param channels: Liste des canaux complexes des 4 utilisateurs.
    :param power_allocation: Liste des puissances allouées (doit être de taille 3).
    :return: Liste des débits maximaux pour chaque utilisateur et la combinaison qui donne le max sum-rate.
    """

    # Puissance totale disponible
    P_total = 10  # en Watts


    num_users = len(channels)
    max_rates = np.zeros(num_users)  # Débits maximaux pour chaque utilisateur
    best_combination = None  # Meilleure combinaison

    # Tester toutes les combinaisons de 3 utilisateurs parmi les 4
    user_combinations = list(combinations(range(num_users), N_users))

    max_sum_rate = -np.inf  # Initialiser avec une valeur très basse

    for comb in user_combinations:
        for Rcomb in list(itertools.permutations(comb, RIS_users)):
            # Sélectionner les canaux des utilisateurs de la combinaison
            selected_channels = [channels[i] for i in comb]
            selected_ris_channel =  [ris_to_users_channels[idx,i] for idx,i in enumerate(Rcomb)]
            # Optimiser l'allocation de puissance
            optimal_power_allocation = optimize_power_allocation(selected_channels, selected_ris_channel, bs_to_ris_channel, P_total,RIS_users)

            # Calculer les capacités des utilisateurs avec cette allocation de puissance optimale
            rates = calculate_noma_capacity(selected_channels, selected_ris_channel, bs_to_ris_channel, optimal_power_allocation,RIS_users)

            # Calculer les débits pour la combinaison
            #rates = calculate_noma_capacity(selected_channels, power_allocation)

            # Calculer le sum-rate pour cette combinaison
            sum_rate = sum(rates)

            # Si le sum-rate est le maximum, mettre à jour les max_rates et la combinaison correspondante
            if sum_rate > max_sum_rate:
                max_sum_rate = sum_rate
                best_combination = comb
                max_rates[comb[0]] = rates[0]
                max_rates[comb[1]] = rates[1]
                max_rates[comb[2]] = rates[2]

    # Créer la combinaison binaire associée de 4 bits
    binary_combination = [0] * num_users  # Initialisation avec tous les zéros
    for user_idx in best_combination:
        binary_combination[user_idx] = 1  # Mettre à 1 les utilisateurs qui participent au NOMA

    binary_combination_str = ''.join(map(str, binary_combination))  # Convertir la liste en string
    binary_combination_bytes = bytes(binary_combination_str, 'utf-8')  # Convertir la chaîne en bytes

    return max_rates, max_sum_rate, binary_combination_bytes


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation des paramètres
    num_realizations = 100000
    noise_power = 1e-9

    N = 5
    N_users = 5
    RIS_users = 3


    # Placer les utilisateurs dans un carré de taille 100x100
    D = 100  # Dimensions de l'espace (zone de placement des utilisateurs)
    bs_position = np.array([D / 2, D / 2])  # Station de base au centre
    
    # Fichier de sortie CSV
    output_file = "train53set-100k.csv"

    
    # Préparer les données pour l'enregistrement
    data = []

    for _ in range(num_realizations):
        ris_positions = np.random.uniform(0, D, (RIS_users, 2))
        bs_to_ris_distance = [np.linalg.norm(bs_position - ris_positions[ris]) for ris in range(RIS_users)]
        if _ % 10000 == 0:
            print(f"Realisation {_}")

        # Placer les utilisateurs aléatoirement dans la cellule (zone de taille D)
        positions = np.random.uniform(0, D, (N, 2))  # Génère N positions (x, y)

        # Calcul des distances entre chaque utilisateur et la station de base
        distances = np.linalg.norm(positions - bs_position, axis=1)  # Calcul des distances euclidiennes

        # Exemple d'atténuation en fonction de la distance (exposant de propagation alpha)
        alpha = 2  # Exposant de propagation
        attenuation = 1 / (distances ** alpha)  # Atténuation en fonction de la distance

        # Génération des canaux complexes (amplitude et phase)
        phases = np.exp(1j * 2 * np.pi * np.random.rand(N))  # Phases aléatoires
        channels = attenuation * phases  # Canaux complexes

        ris_to_users_channels = np.zeros((RIS_users,N), dtype=complex)
        bs_to_ris_channel = np.zeros(RIS_users, dtype=complex)
        for ris in range(RIS_users):
            bs_to_ris_channel[ris] = (1 / (bs_to_ris_distance[ris] ** alpha)) * (np.exp(1j * 2 * np.pi * np.random.rand())) 
            user_to_ris_distance = np.linalg.norm(positions - ris_positions[ris]) 
            ris_to_users_channels[ris] = (1 / (user_to_ris_distance ** alpha)) * (np.exp(1j * 2 * np.pi * np.random.rand(N))) 






        # Générer des canaux aléatoires pour 4 utilisateurs (valeurs complexes)
        #channels = np.random.randn(N) + 1j * np.random.randn(N)

        # Calculer les débits maximaux, sum-rate et la combinaison binaire
        max_rates, max_sum_rate, binary_combination = noma_allocation(channels,bs_to_ris_channel,ris_to_users_channels, N,RIS_users,N_users)

        # Ajouter une ligne au tableau de données avec les résultats
        data.append([
            *[getattr(channel, attr) for channel in bs_to_ris_channel[:RIS_users] for attr in ['real', 'imag']],
            *[getattr(channel, attr) for channel in channels[:N] for attr in ['real', 'imag']],
            *[getattr(channel, attr) for row in ris_to_users_channels for channel in row for attr in ['real', 'imag']],
            max_sum_rate  # Ne pas inclure la binary_combination
        ])

    # Écrire les données dans un fichier CSV
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Écrire l'en-tête
        header = []

        # Générer les colonnes pour chaque utilisateur (chaque canal)
        for i in range(1, RIS_users + 1):
            header.append(f"BS RIS{i}_real")
            header.append(f"BS RIS{i}_imag")

        for i in range(1, N + 1):
            header.append(f"CSI user{i}_real")
            header.append(f"CSI user{i}_imag")

        for ris in range(1, RIS_users + 1):
            for i in range(1, N + 1):
                header.append(f"RIS user{ris}_{i}_real")
                header.append(f"RIS user{ris}_{i}_imag")

        # Ajouter la colonne pour max sum-rate (sans la combinaison binaire)
        header.append("sum-rate max")

        # Écrire l'en-tête dans le fichier CSV
        writer.writerow(header)

        # Écrire les données
        writer.writerows(data)

    print(f"Résultats enregistrés dans {output_file}")
