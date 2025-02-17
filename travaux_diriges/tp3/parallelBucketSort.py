#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Configuration
    N = 1000
    np.random.seed(42)

    # Métriques par processus
    phases = {
        "gen_data": 0.0,
        "scatter": 0.0,
        "local_sort": 0.0,
        "sampling": 0.0,
        "compute_bounds": 0.0,
        "redistribute": 0.0,
        "final_sort": 0.0,
        "gather": 0.0,
    }

    # Calcul des tailles locales et déplacements
    if rank == 0:
        print("N = "+str(N))
        data = np.random.rand(N)
        # Déterminer la taille locale pour chaque processus
        local_sizes = [N // size + 1 if i < N % size else N // size for i in range(size)]
        displacements = np.insert(np.cumsum(local_sizes[:-1]), 0, 0).astype(int)
        start_time = time.time()  # Début de la phase gen_data
    else:
        data, local_sizes, displacements = None, None, None

    # Broadcast des tailles locales à tous les processus
    local_sizes = comm.bcast(local_sizes, root=0)
    local_size = local_sizes[rank]
    local_data = np.empty(local_size, dtype=np.float64)

    # Scatterv pour distribuer les données
    start_time = time.time()
    comm.Scatterv([data, local_sizes, displacements, MPI.DOUBLE], local_data, root=0)
    phases["scatter"] = time.time() - start_time

    # Phase 2: Tri local
    start_time = time.time()
    local_data.sort()
    phases["local_sort"] = time.time() - start_time

    # Phase 3: Échantillonnage pour les bornes
    start_time = time.time()
    if len(local_data) > 0:
        samples = np.linspace(0, len(local_data)-1, size+1, dtype=int)
        local_samples = local_data[samples[1:-1]]
    else:
        local_samples = np.array([])
    all_samples = comm.gather(local_samples, root=0)

    # Calcul des bornes sur le processus 0
    if rank == 0:
        all_samples = np.concatenate(all_samples)
        all_samples.sort()
        bounds = np.linspace(0, len(all_samples), size+1, dtype=int)
        bucket_bounds = all_samples[bounds[1:-1]]
    else:
        bucket_bounds = None

    # Diffusion des bornes à tous les processus
    bucket_bounds = comm.bcast(bucket_bounds, root=0)
    phases["sampling"] = time.time() - start_time

    # Phase 4: Redistribution des données
    start_time = time.time()
    counts = np.zeros(size, dtype=int)
    if len(local_data) > 0:
        indices = np.searchsorted(bucket_bounds, local_data)
        for i in range(size):
            counts[i] = np.sum(indices == i)

    # Échange des données entre processus
    send_counts = comm.alltoall(counts)
    send_buffer = np.empty(np.sum(send_counts), dtype=np.float64)
    displs = np.insert(np.cumsum(counts), 0, 0)[:-1]
    comm.Alltoallv([local_data, counts, displs, MPI.DOUBLE],
                  [send_buffer, send_counts, None, MPI.DOUBLE])
    phases["redistribute"] = time.time() - start_time

    # Phase 5: Tri final et collecte
    start_time = time.time()
    send_buffer.sort()
    gathered_data = comm.gather(send_buffer, root=0)
    phases["final_sort"] = time.time() - start_time

    # Phase 6: Rassemblement sur le processus 0
    start_time = time.time()
    if rank == 0:
        sorted_data = np.concatenate(gathered_data)
        phases["gather"] = time.time() - start_time
        assert np.all(np.diff(sorted_data) >= 0), "Erreur de tri"
    else:
        sorted_data = None

    # Collecte des métriques par processus
    all_phases = comm.gather(phases, root=0)

    # Affichage des métriques sur le processus 0
    if rank == 0:
        print("Métriques par processus :")
        for i, process_phases in enumerate(all_phases):
            print(f"Processus {i}:")
            for phase, duration in process_phases.items():
                print(f"  {phase}: {duration:.6f} secondes")
            print()
        #print("sorted data : ",sorted_data)

if __name__ == "__main__":
    main()