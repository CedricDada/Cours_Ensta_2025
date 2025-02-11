#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

# Initialisation de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp  = comm.Get_size()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    # La méthode __contains__ n'est pas utilisée ici, on peut l'omettre ou la corriger.
    # def __contains__(self, c: complex) -> bool:
    #     return self.stability(c) == 1

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        # Vérifications rapides d’appartenance à des zones connues
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
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
                    return iter + 1 - log(log(abs(z))) / log(2)
                return iter
        return self.max_iterations

# Paramètres du calcul
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3.0 / width
scaleY = 2.25 / height

# ================================
# Répartition cyclique (round-robin)
# ================================
# Chaque processus prend les lignes globales dont l'indice y satisfait : y % nbp == rank
local_rows = [y for y in range(height) if y % nbp == rank]
local_height = len(local_rows)

# On alloue un tableau pour stocker le résultat local
convergence_local = np.empty((local_height, width), dtype=np.double)

# Calcul des lignes attribuées
deb_local = time()
for local_idx, y_global in enumerate(local_rows):
    for x in range(width):
        c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y_global)
        convergence_local[local_idx, x] = mandelbrot_set.convergence(c, smooth=True)
fin_local = time()
duration_local = fin_local - deb_local

# Récupération des temps de calcul sur le processus 0
durations = comm.gather(duration_local, root=0)

# ================================
# Rassemblement des données via MPI_Gatherv
# ================================
# Il faut aussi rassembler la liste des indices globaux pour pouvoir reconstituer l'image
all_local_rows = comm.gather(local_rows, root=0)
local_heights = comm.gather(local_height, root=0)  # nombre de lignes par processus

# Préparation du sendbuf (les données sont aplaties)
sendbuf = convergence_local.ravel()

if rank == 0:
    # Pour chaque processus, le nombre d'éléments envoyés = (nombre de lignes) * width
    recv_counts = [int(h * width) for h in local_heights]
    # Calcul des déplacements en s'assurant d'obtenir des entiers
    displacements = np.insert(np.cumsum(recv_counts[:-1], dtype=int), 0, 0)
    # Allocation d'un tampon temporaire pour recueillir toutes les lignes
    gathered_flat = np.empty(sum(recv_counts), dtype=np.double)
    recvbuf = (gathered_flat, recv_counts, displacements, MPI.DOUBLE)
else:
    recvbuf = None


comm.Gatherv(sendbuf, recvbuf, root=0)

# Reconstruction de l'image finale sur le processus 0
if rank == 0:
    # On découpe le tampon rassemblé en un bloc par processus
    blocks = []
    start_idx = 0
    for count in recv_counts:
        block = gathered_flat[start_idx:start_idx + count].reshape(-1, width)
        blocks.append(block)
        start_idx += count

    # On alloue le tableau final
    convergence = np.empty((height, width), dtype=np.double)
    # Pour chaque processus, on place chaque ligne dans sa position globale
    for proc in range(nbp):
        for local_idx, global_row in enumerate(all_local_rows[proc]):
            convergence[global_row, :] = blocks[proc][local_idx, :]

    # Affichage des résultats
    print("Matrice de convergence (forme):", convergence.shape)
    print("\nTemps de calcul par processus (secondes) :")
    for i, d in enumerate(durations):
        print(f"  - Processus {i}: {d:.3f}")
    max_time = max(durations)
    min_time = min(durations)
    imbalance_ratio = (max_time - min_time) / max_time * 100
    print(f"\nDéséquilibre max/min : {max_time:.3f} s vs {min_time:.3f} s")
    print(f"Ratio de déséquilibrage : {imbalance_ratio:.1f}%")
    
    # Génération de l'image
    deb_img = time()
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    fin_img = time()
    print(f"\nTemps de constitution de l'image : {fin_img - deb_img:.3f} s")
    image.show()
