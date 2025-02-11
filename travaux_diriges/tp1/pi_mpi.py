
#!/usr/bin/env python3
# pi_mpi.py
from mpi4py import MPI
import time, numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nb_samples_total = 40_000_000
nb_samples = nb_samples_total // size  # repartition uniforme des echantillons

# Debut de la mesure de temps (commence avant le calcul)
start = MPI.Wtime()

# Chaque processus initialise sa graine pour generer des nombres aleatoires differents
np.random.seed(int(time.time()) ^ rank)

# Generation des points aleatoires dans [-1, 1] pour x et y
x = np.random.uniform(-1, 1, nb_samples)
y = np.random.uniform(-1, 1, nb_samples)

# Compter le nombre de points tombant dans le cercle unite
local_count = np.count_nonzero(x*x + y*y <= 1.0)

# Reduire les resultats de tous les processus (somme)
total_count = comm.reduce(local_count, op=MPI.SUM, root=0)

# Fin de la mesure de temps
end = MPI.Wtime()

# Le processus maitre calcule l'approximation de pi et affiche le temps d'execution
if rank == 0:
    approx_pi = 4.0 * total_count / nb_samples_total
    print(f"Approximation de pi (mpi4py) = {approx_pi}")
    print(f"Temps d'execution = {end - start} secondes")