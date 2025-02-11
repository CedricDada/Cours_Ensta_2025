# Produit matrice-vecteur v = A.u
import numpy as np
import time  # Import pour mesurer le temps

# Dimension du problème (peut-être changé)
dim = 120

# Initialisation de la matrice
A = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(dim)])
print(f"A = {A}")

# Initialisation du vecteur u
u = np.array([i + 1. for i in range(dim)])
print(f"u = {u}")

# Mesure du temps avant le calcul
start_time = time.time()

# Produit matrice-vecteur
v = A.dot(u)

# Mesure du temps après le calcul
end_time = time.time()

# Calcul de la durée du calcul
calc_time = end_time - start_time

# Affichage du résultat et du temps de calcul
print(f"v = {v}")
print(f"Temps de calcul du produit matrice-vecteur : {calc_time:.6f} secondes")