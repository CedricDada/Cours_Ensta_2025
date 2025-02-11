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
nbp = comm.Get_size()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

    def convergence(self, c: complex, smooth=False, clamp=True) -> float:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return max(0.0, min(value, 1.0)) if clamp else value

    def count_iterations(self, c: complex, smooth=False) -> int | float:
        # Vérifications rapides d'appartenance à des zones connues
        if c.real * c.real + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real + 1) * (c.real + 1) + c.imag * c.imag < 0.0625:
            return self.max_iterations
        if (c.real > -0.75) and (c.real < 0.5):
            ct = c.real - 0.25 + 1.j * c.imag
            ctnrm2 = abs(ct)
            if ctnrm2 < 0.5 * (1 - ct.real / max(ctnrm2, 1.E-14)):
                return self.max_iterations
        # Boucle d'itération
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

if rank == 0:
    # Maître : distribue les lignes
    convergence = np.empty((height, width), dtype=np.double)
    next_row = 0
    active_workers = nbp - 1
    workers_done = 0
    results_received = 0
    total_durations = {}
    lines_calculated = {}

    start_time = time()
    print(f"[Maître] Début de la distribution des tâches avec {active_workers} workers")

    # Distribution initiale des tâches
    for worker in range(1, nbp):
        if next_row < height:
            comm.send(next_row, dest=worker, tag=1)
            next_row += 1
        else:
            comm.send(None, dest=worker, tag=2)
            workers_done += 1

    # Boucle principale
    while results_received < height:
        # Recevoir un résultat
        status = MPI.Status()
        y_global, conv_slice = comm.recv(source=MPI.ANY_SOURCE, tag=3, status=status)
        source = status.Get_source()
        
        # Stocker le résultat
        convergence[y_global, :] = conv_slice
        results_received += 1

        # Envoyer plus de travail si disponible
        if next_row < height:
            comm.send(next_row, dest=source, tag=1)
            next_row += 1
        else:
            comm.send(None, dest=source, tag=2)
            workers_done += 1

    # Recevoir les statistiques finales
    for i in range(1, nbp):
        total_duration = comm.recv(source=i, tag=4)
        calculated_lines = comm.recv(source=i, tag=5)
        total_durations[i] = total_duration
        lines_calculated[i] = calculated_lines

    end_time = time()
    total_duration_master = end_time - start_time

    # Génération de l'image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(convergence) * 255))
    image.show()

    # Affichage des statistiques
    print("\nTemps total de calcul par processus (secondes) :")
    for source, duration in total_durations.items():
        print(f"  - Processus {source}: {duration:.3f}")

    if total_durations:
        max_time = max(total_durations.values())
        min_time = min(total_durations.values())
        imbalance_ratio = (max_time - min_time) / max_time * 100
        print(f"\nDéséquilibre max/min : {max_time:.3f} s vs {min_time:.3f} s")
        print(f"Ratio de déséquilibrage : {imbalance_ratio:.1f}%")
        print(f"Durée totale effective du calcul parallèle : {max_time:.3f} s")

    print("\nNombre de Lignes calculées par chaque processus :")
    for source, lines in lines_calculated.items():
        print(f"  - Processus {source}: {len(lines)}")

    print(f"\nDurée totale d'exécution (incluant les communications) : {total_duration_master:.3f} s")

else:
    # Esclave : calcule les lignes assignées
    total_duration = 0
    calculated_lines = []

    while True:
        # Recevoir une tâche
        y_global = comm.recv(source=0, tag=MPI.ANY_TAG)
        
        if y_global is None:  # Plus de travail
            break

        # Calcul de la ligne
        conv_slice = np.empty(width, dtype=np.double)
        deb_local = time()
        for x in range(width):
            c = complex(-2.0 + scaleX * x, -1.125 + scaleY * y_global)
            conv_slice[x] = mandelbrot_set.convergence(c, smooth=True)
        fin_local = time()
        
        # Mise à jour des statistiques
        duration_local = fin_local - deb_local
        total_duration += duration_local
        calculated_lines.append(y_global)
        
        # Envoyer le résultat
        comm.send((y_global, conv_slice), dest=0, tag=3)

    # Envoyer les statistiques finales
    comm.send(total_duration, dest=0, tag=4)
    comm.send(calculated_lines, dest=0, tag=5)