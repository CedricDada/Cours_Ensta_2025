#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

# Initialisation de MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
nbp    = comm.Get_size()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        # Vérifications rapides d’appartenance à des zones connues :
        if c.real*c.real + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real+1)*(c.real+1) + c.imag*c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Boucle d’itération
        z = 0
        for iter in range(self.max_iterations):
            z = z*z + c
            if abs(z) > self.escape_radius:
                if smooth:
                    return iter + 1 - log(log(abs(z)))/log(2)
                return iter
        return self.max_iterations

# Paramètres du calcul
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3.0 / width
scaleY = 2.25 / height

# Découpage de l’image par lignes
chunk = height // nbp
rest  = height % nbp
start = rank * chunk + min(rank, rest)
end   = start + chunk + (1 if rank < rest else 0)
local_height = end - start

# Pour faciliter le rassemblement, on travaille avec des tableaux dont les lignes sont contiguës :
# Chaque processus calcule un tableau de forme (local_height, width)
convergence_local = np.empty((local_height, width), dtype=np.double)

# Mesure du temps local et calcul du Mandelbrot sur la tranche assignée
deb_local = time()
for y_local in range(local_height):
    y_global = start + y_local
    for x in range(width):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y_global)
        convergence_local[y_local, x] = mandelbrot_set.convergence(c, smooth=True)
fin_local = time()
duration_local = fin_local - deb_local

# Récupération des temps de calcul sur le processus 0
durations = comm.gather(duration_local, root=0)

# Rassemblement des tableaux de convergence via MPI_Gatherv
# Chaque processus envoie un bloc de taille (local_height * width)
sendbuf = convergence_local  # Il est contigu en mémoire (C-order)

# On rassemble d’abord la taille (en nombre de lignes) de chaque bloc sur le processus 0
local_heights = comm.gather(local_height, root=0)

if rank == 0:
    # Pour chaque processus, le nombre d'éléments envoyés = (local_height * width)
    recv_counts = np.array([h * width for h in local_heights], dtype=int)
    # Calcul des déplacements (en nombre d'éléments) dans le tableau global
    displacements = np.insert(np.cumsum(recv_counts[:-1]), 0, 0)
    # Allocation du tableau global qui rassemblera toutes les lignes (forme : (height, width))
    convergence = np.empty((height, width), dtype=np.double)
    # On travaille sur la vue plate du tableau
    recvbuf = (convergence.ravel(), recv_counts, displacements, MPI.DOUBLE)
else:
    recvbuf = None

comm.Gatherv(sendbuf=sendbuf, recvbuf=recvbuf, root=0)

# Affichage des résultats sur le processus 0
if rank == 0:
    print("Matrice de convergence (forme):", convergence.shape)
    print("\nTemps de calcul par processus (secondes) :")
    for i, d in enumerate(durations):
        print(f"  - Processus {i}: {d:.3f}")
    max_time = max(durations)
    min_time = min(durations)
    imbalance_ratio = (max_time - min_time) / max_time * 100
    print(f"\nDéséquilibre max/min : {max_time:.3f} s vs {min_time:.3f} s")
    print(f"Ratio de déséquilibrage : {imbalance_ratio:.1f}%")
    
    # Génération de l'image (les lignes sont déjà dans l'ordre)
    deb_img = time()
    # On applique une colormap, on passe dans uint8 et on crée l'image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    fin_img = time()
    print(f"\nTemps de constitution de l'image : {fin_img - deb_img:.3f} s")
    image.show()
