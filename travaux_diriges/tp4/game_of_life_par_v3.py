#!/usr/bin/env python3
import pygame as pg
import numpy as np
from mpi4py import MPI
import sys, time

ITERATIONS = 50 # Itérations par défaut

# Duplication du communicateur global et séparation des processus
globCom = MPI.COMM_WORLD.Dup()
rank = globCom.Get_rank()
nbp  = globCom.Get_size()
# Les processus esclaves forment un communicateur dédié
newCom = globCom.Split(rank != 0, rank)
if rank == 0:
    print(f"[Master] Rang global : {rank}")
else:
    print(f"[Slave {newCom.Get_rank()}] Rang global : {rank}, Rang local : {newCom.Get_rank()}, nb locaux : {newCom.Get_size()}")

class Grille:
    """
    Grille torique locale avec ghost cells.
    La grille globale est découpée sur les processus esclaves.
    """
    def __init__(self, rank: int, nbp: int, dim, init_pattern=None,
                 color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dim
        # Dimensions locales (sans ghost cells)
        self.dimensions_loc = (dim[0]//nbp + (1 if rank < dim[0] % nbp else 0), dim[1])
        # Position de départ dans la grille globale
        self.start_loc = rank * self.dimensions_loc[0] + (dim[0] % nbp if rank >= dim[0] % nbp else 0)
        # Création de la grille locale avec 2 lignes ghost (en haut et en bas)
        if init_pattern is not None:
            self.cells = np.zeros((self.dimensions_loc[0] + 2, self.dimensions_loc[1]), dtype=np.uint8)
            indices_i = [v[0] - self.start_loc + 1 for v in init_pattern
                         if self.start_loc <= v[0] < self.start_loc + self.dimensions_loc[0]]
            indices_j = [v[1] for v in init_pattern]
            if indices_i:
                self.cells[indices_i, indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=(self.dimensions_loc[0] + 2, self.dimensions_loc[1]), dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next_iteration(self):
        """
        Calcule la prochaine génération.
        Retourne le diff (non utilisé ici) et le temps de calcul.
        """
        t0 = time.time()
        neighbours = sum(
            np.roll(np.roll(self.cells, i, 0), j, 1)
            for i in (-1, 0, 1) for j in (-1, 0, 1)
            if (i, j) != (0, 0)
        )
        next_cells = (neighbours == 3) | (self.cells & (neighbours == 2))
        diff = (next_cells != self.cells)
        # Conversion explicite pour obtenir des 0/1
        self.cells = next_cells.astype(np.uint8)
        return diff, time.time() - t0

    def update_ghost_cells(self):
        """
        Échange les ghost cells entre processus esclaves.
        Retourne le temps écoulé.
        """
        t0 = time.time()
        req1 = newCom.Irecv(self.cells[-1, :],
                            source=(newCom.Get_rank() + 1) % newCom.Get_size(), tag=101)
        req2 = newCom.Irecv(self.cells[0, :],
                            source=(newCom.Get_rank() - 1) % newCom.Get_size(), tag=102)
        newCom.Send(self.cells[-2, :],
                    dest=(newCom.Get_rank() + 1) % newCom.Get_size(), tag=102)
        newCom.Send(self.cells[1, :],
                    dest=(newCom.Get_rank() - 1) % newCom.Get_size(), tag=101)
        req1.Wait()
        req2.Wait()
        return time.time() - t0

class App:
    """
    Application d'affichage du jeu de la vie.
    Le rendu est basé sur la grille globale (sans ghost cells).
    """
    def __init__(self, geometry, grid):
        self.grid = grid
        self.size_x = geometry[1] // grid.dimensions[1]
        self.size_y = geometry[0] // grid.dimensions[0]
        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))
        self.draw_color = pg.Color('lightgrey') if self.size_x > 4 and self.size_y > 4 else None
        # Extraction des composantes RGB (sans alpha)
        self.colors = np.array([self.grid.col_dead[:-1], self.grid.col_life[:-1]])

    def draw(self):
        t0 = time.time()
        # Rendu de la grille globale (on retire les ghost cells)
        surface = pg.surfarray.make_surface(self.colors[self.grid.cells[1:-1, :].T])
        surface = pg.transform.flip(surface, False, True)
        surface = pg.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))
        if self.draw_color is not None:
            for i in range(self.grid.dimensions[0]):
                pg.draw.line(self.screen, self.draw_color, (0, i*self.size_y), (self.width, i*self.size_y))
            for j in range(self.grid.dimensions[1]):
                pg.draw.line(self.screen, self.draw_color, (j*self.size_x, 0), (j*self.size_x, self.height))
        pg.display.update()
        return time.time() - t0

if __name__ == '__main__':
    import sys, time, pygame as pg
    # Dictionnaire des motifs
    dico_patterns = {
        'blinker': ((5,5),[(2,1),(2,2),(2,3)]),
        'toad': ((6,6),[(2,2),(2,3),(2,4),(3,3),(3,4),(3,5)]),
        "acorn": ((100,100), [(51,52),(52,54),(53,51),(53,52),(53,55),(53,56),(53,57)]),
        "beacon": ((6,6), [(1,3),(1,4),(2,3),(2,4),(3,1),(3,2),(4,1),(4,2)]),
        "boat": ((5,5),[(1,1),(1,2),(2,1),(2,3),(3,2)]),
        "glider": ((100,90),[(1,1),(2,2),(2,3),(3,1),(3,2)]),
        "glider_gun": ((200,100),[(51,76),(52,74),(52,76),(53,64),(53,65),
                                   (53,72),(53,73),(53,86),(53,87),(54,63),
                                   (54,67),(54,72),(54,73),(54,86),(54,87),
                                   (55,52),(55,53),(55,62),(55,68),(55,72),
                                   (55,73),(56,52),(56,53),(56,62),(56,66),
                                   (56,68),(56,69),(56,74),(56,76),(57,62),
                                   (57,68),(57,76),(58,63),(58,67),(59,64),
                                   (59,65)])
    }
    # Lecture des arguments
    choice = 'glider'
    resx, resy = 800, 800
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    if len(sys.argv) > 3:
        resx = int(sys.argv[2])
        resy = int(sys.argv[3])
    if len(sys.argv) > 4:
        ITERATIONS = int(sys.argv[4])
    print(f"Pattern initial choisi : {choice}")
    print(f"résolution écran : {(resx, resy)}")
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("Motif inconnu. Options :", list(dico_patterns.keys()))
        sys.exit(1)
    
    # Processus maître
    if rank == 0:
        pg.init()
        # Pour le maître, on crée une grille globale (non découpée)
        grid = Grille(0, 1, *init_pattern)
        appli = App((resx, resy), grid)
        iter_count = 0
        while iter_count < ITERATIONS:
            t_comm = time.time()
            # Envoi du signal de calcul au processus esclave
            globCom.send(1, dest=1)
            # Réception de la grille globale calculée (attendue du global rank 1)
            grid_data = globCom.recv(source=1)
            comm_time = time.time() - t_comm
            # Mise à jour de la grille globale (hors ghost cells)
            appli.grid.cells[1:-1, :] = grid_data
            t_rend = time.time()
            appli.draw()
            rend_time = time.time() - t_rend
            iter_count += 1
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    iter_count = ITERATIONS
                    pg.quit()
                    globCom.send(-1, dest=1)
            print(f"[Master] It {iter_count:03d}: Comm {comm_time:2.2e}s, Rendu {rend_time:2.2e}s", flush=True)
        pg.quit()
    # Processus esclave
    else:
        grid = Grille(newCom.Get_rank(), newCom.Get_size(), *init_pattern)
        ghost_init = grid.update_ghost_cells()
        print(f"[Slave {newCom.Get_rank()}] Initial ghost update : {ghost_init:2.2e}s")
        grid_glob = None
        if newCom.Get_rank() == 0:
            grid_glob = np.zeros(init_pattern[0], dtype=np.uint8)
        sendcounts = np.array(newCom.gather(grid.cells[1:-1, :].size, root=0))
        iter_count = 0
        while iter_count < ITERATIONS:
            t_total = time.time()
            # Calcul de la prochaine génération
            _, comp_time = grid.compute_next_iteration()
            # Mise à jour des ghost cells
            ghost_time = grid.update_ghost_cells()
            # Communication : rassemblement global via Gatherv
            t_gatherv = time.time()
            newCom.Gatherv(grid.cells[1:-1, :], [grid_glob, sendcounts], root=0)
            gatherv_time = time.time() - t_gatherv
            # Temps de communication total = ghost update + Gatherv
            comm_time = ghost_time + gatherv_time
            iter_count += 1
            # Le processus esclave local 0 envoie la grille globale au maître
            if newCom.Get_rank() == 0:
                if globCom.Iprobe(source=0):
                    msg = globCom.recv(source=0)
                    if msg == -1:
                        break
                globCom.send(grid_glob, dest=0)
            total_time = time.time() - t_total
            print(f"[Slave {newCom.Get_rank()}] It {iter_count:03d}: Calcul {comp_time:2.2e}s, Ghost {ghost_time:2.2e}s, Comm {comm_time:2.2e}s, Total {total_time:2.2e}s", flush=True)
       
