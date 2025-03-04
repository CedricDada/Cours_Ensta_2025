import pygame as pg
import numpy as np
from mpi4py import MPI
import sys
import time

ITERATIONS = 50

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
    def __init__(self, dim, init_pattern=None, color_life=pg.Color("black"), color_dead=pg.Color("white")):
        self.dimensions = dim
        if init_pattern is not None:
            self.cells = np.zeros(self.dimensions, dtype=np.uint8)
            indices_i = [v[0] for v in init_pattern]
            indices_j = [v[1] for v in init_pattern]
            self.cells[indices_i, indices_j] = 1
        else:
            self.cells = np.random.randint(2, size=dim, dtype=np.uint8)
        self.col_life = color_life
        self.col_dead = color_dead

    def compute_next(self):
        t0 = time.time()
        neighbours = sum(np.roll(np.roll(self.cells, i, 0), j, 1)
                         for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
        next_cells = (neighbours == 3) | (self.cells & (neighbours == 2))
        self.cells = next_cells.astype(np.uint8)
        return self.cells, (time.time() - t0)

class App:
    def __init__(self, geometry, grid):
        self.grid = grid
        self.size_x = geometry[0] // grid.dimensions[1]
        self.size_y = geometry[1] // grid.dimensions[0]
        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))
        self.col_dead = np.array(self.grid.col_dead[:-1])
        self.col_life = np.array(self.grid.col_life[:-1])
        self.draw_color = pg.Color('lightgrey') if (self.size_x > 4 and self.size_y > 4) else None

    def draw(self, cells):
        t0 = time.time()
        rgb_array = np.zeros((cells.shape[1], cells.shape[0], 3), dtype=np.uint8)
        rgb_array[cells.T == 0] = self.col_dead
        rgb_array[cells.T == 1] = self.col_life
        surface = pg.surfarray.make_surface(rgb_array)
        surface = pg.transform.flip(surface, False, True)
        surface = pg.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))
        if self.draw_color:
            for i in range(self.grid.dimensions[0]):
                pg.draw.line(self.screen, self.draw_color, (0, i * self.size_y), (self.width, i * self.size_y))
            for j in range(self.grid.dimensions[1]):
                pg.draw.line(self.screen, self.draw_color, (j * self.size_x, 0), (j * self.size_x, self.height))
        pg.display.update()
        return (time.time() - t0)

if __name__ == "__main__":
    choice = 'glider' if len(sys.argv) < 2 else sys.argv[1]
    resx, resy = (800, 800) if len(sys.argv) < 4 else (int(sys.argv[2]), int(sys.argv[3]))
    try:
        init_pattern = dico_patterns[choice]
    except KeyError:
        print("Pattern inconnu. Options :", list(dico_patterns.keys()))
        sys.exit(1)

    if rank == 0:
        pg.init()
        grid = Grille(*init_pattern)
        app = App((resx, resy), grid)
        iter_count = 0
        recv_req = comm.irecv(source=1, tag=0)
        while iter_count < ITERATIONS:
            flag, data = recv_req.test()
            if flag:
                cells, comp_time = data
                render_time = app.draw(cells)
                iter_count += 1
                print(f"[Master] It {iter_count:03d}: Calcul {comp_time:.4e}s, Rendu {render_time:.4e}s")
                recv_req = comm.irecv(source=1, tag=0)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    comm.send(True, dest=1, tag=1)
                    pg.quit()
                    sys.exit(0)
        comm.send(True, dest=1, tag=1)
        pg.quit()
    else:
        grid = Grille(*init_pattern)
        iter_count = 0
        send_req = None
        while True:
            cells, comp_time = grid.compute_next()
            iter_count += 1
            t_comm = time.time()
            if send_req:
                send_req.Wait()
            send_req = comm.isend((cells, comp_time), dest=0, tag=0)
            comm_time = time.time() - t_comm
            print(f"[Slave] It {iter_count:03d}: Calcul {comp_time:.4e}s, Comm {comm_time:.4e}s")
            if comm.Iprobe(source=0, tag=1):
                comm.recv(source=0, tag=1)
                break
