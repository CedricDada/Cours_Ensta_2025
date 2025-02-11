# matvec_col_mpi.py
from mpi4py import MPI
import numpy as np
import time  # Import pour mesurer le temps

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nbp = comm.Get_size()

dim = 120  # Doit être divisible par nbp
Nloc = dim // nbp

local_calc_start = time.time()  # Début du calcul local
# Initialisation locale
A_local = np.array([[(i + j) % dim + 1. for i in range(rank * Nloc, (rank + 1) * Nloc)] for j in range(dim)])
u = np.array([i + 1. for i in range(dim)])

# Broadcast de u à tous
u = comm.bcast(u, root=0)

# Calcul local avec chronométrage
v_local = A_local.dot(u[rank * Nloc:(rank + 1) * Nloc])
local_calc_time = time.time() - local_calc_start  # Fin du calcul local

# Rassemblement avec somme
v = np.zeros(dim, dtype=np.float64)
comm.Allreduce(v_local, v, op=MPI.SUM)

# Envoi du temps de calcul local au processus 0
if rank != 0:
    comm.send(local_calc_time, dest=0)

# Collecte des durées de calcul local sur le processus 0 et affichage
if rank == 0:
    all_times = [local_calc_time]  # Temps du processus 0
    for i in range(1, nbp):
        calc_time = comm.recv(source=i)
        all_times.append(calc_time)

    # Affichage des durées de calcul local pour tous les processus
    for i, calc_time in enumerate(all_times):
        print(f"Processus {i}: Temps de calcul local : {calc_time:.6f} secondes")