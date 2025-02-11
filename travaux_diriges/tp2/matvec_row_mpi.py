# matvec_row_mpi.py
from mpi4py import MPI
import numpy as np
import time  # Pour mesurer les temps d'exécution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

dim = 120
Nloc = dim // nbp

# Initialisation locale
A_local = np.array([[(i + j) % dim + 1. for i in range(dim)] for j in range(rank * Nloc, (rank + 1) * Nloc)])
u = np.array([i + 1. for i in range(dim)])

# Broadcast de u à tous
u = comm.bcast(u, root=0)

# Calcul local avec chronométrage
calc_start = time.time()
v_local = A_local.dot(u)
calc_time = time.time() - calc_start

# Rassemblement
v = np.zeros(dim, dtype=np.float64)
comm.Allgather(v_local, v)

# Envoi du temps de calcul local au processus 0
if rank != 0:
    comm.send(calc_time, dest=0)

# Collecte des temps de calcul local par le processus 0
if rank == 0:
    all_calc_times = [calc_time]  # Temps de calcul local du processus 0
    for i in range(1, nbp):
        calc_time = comm.recv(source=i)
        all_calc_times.append(calc_time)

    # Affichage des temps de calcul local pour chaque processus
    for i, calc_t in enumerate(all_calc_times):
        print(f"Processus {i} : Temps de calcul local : {calc_t:.6f} secondes")

    #print(f"\nRésultat final (processus 0) : v = {v}")