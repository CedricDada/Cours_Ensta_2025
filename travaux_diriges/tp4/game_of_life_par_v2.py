#!/usr/bin/env python3
import pygame as pg
import numpy as np
from mpi4py import MPI
import sys
import time

# Nombre d'itérations par défaut
ITERATIONS = 50
BATCH_SIZE = 10  

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

dico_patterns = {
    'blinker': ((5, 5), [(2, 1), (2, 2), (2, 3)]),
    'toad': ((6, 6), [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5)]),
    "acorn": ((100, 100), [(51, 52), (52, 54), (53, 51), (53, 52), (53, 55), (53, 56), (53, 57)]),
    "beacon": ((6, 6), [(1, 3), (1, 4), (2, 3), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2)]),
    "glider": ((100, 90), [(1, 1), (2, 2), (2, 3), (3, 1), (3, 2)]),
    "glider_gun": ((200, 100), [(51, 76), (52, 74), (52, 76), (53, 64), (53, 65),
                                (53, 72), (53, 73), (53, 86), (53, 87), (54, 63),
                                (54, 67), (54, 72), (54, 73), (54, 86), (54, 87),
                                (55, 52), (55, 53), (55, 62), (55, 68), (55, 72),
                                (55, 73), (56, 52), (56, 53), (56, 62), (56, 66),
                                (56, 68), (56, 69), (56, 74), (56, 76), (57, 62),
                                (57, 68), (57, 76), (58, 63), (58, 67), (59, 64),
                                (59, 65)])
}

class Grille:
    def __init__(self, pattern_key):
        self.dim, pattern = dico_patterns[pattern_key]
        self.cells = np.zeros(self.dim, dtype=np.uint8)
        rows, cols = zip(*pattern)
        self.cells[rows, cols] = 1

    def compute_next(self, next_gen=None):
        if next_gen is None:
            next_gen = np.empty_like(self.cells)
        t0 = time.time()
        neighbours = sum(
            np.roll(np.roll(self.cells, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i, j) != (0, 0)
        )
        result = ((neighbours == 3) | (self.cells & (neighbours == 2))).astype(np.uint8)
        next_gen[:, :] = result
        comp_time = time.time() - t0
        return next_gen, comp_time
    def compute_batch(self, batch_buffer):
        """Calcule un lot de BATCH_SIZE itérations."""
        current = self.cells.copy()
        for i in range(BATCH_SIZE):
            next_grid, _ = self.compute_next(current)
            batch_buffer[i] = next_grid
            current = next_grid
        self.cells = current  
        return batch_buffer

class Actualisation:
    def __init__(self, grid_dim, resolution=(800, 800)):
        pg.init()
        self.grid_size = grid_dim
        self.resolution = resolution
        self.cell_w = resolution[0] // grid_dim[1]
        self.cell_h = resolution[1] // grid_dim[0]
        self.width = grid_dim[1] * self.cell_w
        self.height = grid_dim[0] * self.cell_h
        self.screen = pg.display.set_mode((self.width, self.height))
        # Tableau de couleurs : index 0 pour cellule morte, 1 pour cellule vivante
        self.colors = np.array([
            pg.Color('white')[:3],
            pg.Color('black')[:3]
        ], dtype=np.uint8)

    def draw_grid(self, cells):
        t0 = time.time()
        cells = (cells != 0).astype(np.uint8)
        surface = pg.surfarray.make_surface(self.colors[cells.T])
        surface = pg.transform.flip(surface, False, True)
        if (self.cell_w, self.cell_h) != (1, 1):
            surface = pg.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))
        if self.cell_w > 4 and self.cell_h > 4:
            color = pg.Color('lightgrey')
            for x in range(0, self.width, self.cell_w):
                pg.draw.line(self.screen, color, (x, 0), (x, self.height))
            for y in range(0, self.height, self.cell_h):
                pg.draw.line(self.screen, color, (0, y), (self.width, y))
        pg.display.update()
        rend_time = time.time() - t0
        return rend_time

def maitre(pattern_key, resolution):
    grid = Grille(pattern_key)
    renderer = Actualisation(grid.dim, resolution)
    batch_shape = (BATCH_SIZE, *grid.cells.shape)
    recv_buffer = np.empty(batch_shape, dtype=np.uint8)
    req_recv = comm.Irecv(recv_buffer, source=1, tag=0)
    iter_count = 0

    while iter_count < ITERATIONS:
        if req_recv.Test():
            # Affiche chaque itération du lot
            for i in range(BATCH_SIZE):
                if iter_count + i >= ITERATIONS:
                    break
                rend_time = renderer.draw_grid(recv_buffer[i])
                print(f"[Master] It {iter_count + i + 1}: Rendu {rend_time:.4e}s")
            iter_count += BATCH_SIZE
            # Envoi ACK pour le lot
            comm.isend(True, dest=1, tag=1)
            req_recv = comm.Irecv(recv_buffer, source=1, tag=0)
        # Gestion fermeture
        for event in pg.event.get():
            if event.type == pg.QUIT:
                comm.isend(True, dest=1, tag=2)
                pg.quit()
                return
    comm.isend(True, dest=1, tag=2)
    pg.quit()

def esclave(pattern_key):
    grid = Grille(pattern_key)
    batch_shape = (BATCH_SIZE, *grid.cells.shape)
    buffers = [np.empty(batch_shape, dtype=np.uint8) for _ in range(2)]
    current_buffer = 0
    req_send = None
    iter_count = 0

    # Premier lot
    t0 = time.time()
    buffers[current_buffer] = grid.compute_batch(buffers[current_buffer])
    batch_time = time.time() - t0
    req_send = comm.Isend(buffers[current_buffer], dest=0, tag=0)
    current_buffer = 1 - current_buffer
    iter_count += BATCH_SIZE
    print(f"[Slave] Lot envoyé (jusqu'à itération {iter_count}), temps de calcul du lot : {batch_time:.4e}s")

    while iter_count < ITERATIONS:
        # Calcul du prochain lot si le buffer est libre
        if req_send is None or req_send.Test():
            t0 = time.time()
            buffers[current_buffer] = grid.compute_batch(buffers[current_buffer])
            batch_time = time.time() - t0
            req_send = comm.Isend(buffers[current_buffer], dest=0, tag=0)
            current_buffer = 1 - current_buffer
            iter_count += BATCH_SIZE
            print(f"[Slave] Lot envoyé (jusqu'à itération {iter_count}), temps de calcul du lot : {batch_time:.4e}s")
        
        # Vérification ACK ou terminaison
        if comm.Iprobe(source=0, tag=1):
            comm.recv(source=0, tag=1)
        if comm.Iprobe(source=0, tag=2):
            comm.recv(source=0, tag=2)
            break

if __name__ == "__main__":
    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: mpirun -n 2 python3 game_of_life_par_v2.py <pattern> [width] [height] [ITERATIONS]")
            print("Available patterns:", list(dico_patterns.keys()))
        sys.exit(1)
    
    pattern_key = sys.argv[1]
    if len(sys.argv) > 3:
        res = (int(sys.argv[2]), int(sys.argv[3]))
    else:
        res = (800, 800)
    if len(sys.argv) > 4:
        ITERATIONS = int(sys.argv[4])
    
    if rank == 0:
        maitre(pattern_key, res)
    else:
        esclave(pattern_key)
