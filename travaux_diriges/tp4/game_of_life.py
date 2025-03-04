#!/usr/bin/env python
import pygame as pg
import numpy as np
import time
import sys

ITERATIONS = 50

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

    def compute_next_iteration(self):
        t0 = time.time()
        neighbours = sum(np.roll(np.roll(self.cells, i, 0), j, 1)
                         for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0))
        next_cells = (neighbours == 3) | (self.cells & (neighbours == 2))
        self.cells = next_cells.astype(np.uint8)
        return (time.time() - t0)

class App:
    def __init__(self, geometry, grid):
        self.grid = grid
        self.size_x = geometry[1] // grid.dimensions[1]
        self.size_y = geometry[0] // grid.dimensions[0]
        self.width = grid.dimensions[1] * self.size_x
        self.height = grid.dimensions[0] * self.size_y
        self.screen = pg.display.set_mode((self.width, self.height))
        self.draw_color = pg.Color('lightgrey') if self.size_x > 4 and self.size_y > 4 else None
        self.colors = np.array([self.grid.col_dead[:-1], self.grid.col_life[:-1]])

    def draw(self):
        t0 = time.time()
        surface = pg.surfarray.make_surface(self.colors[self.grid.cells.T])
        surface = pg.transform.flip(surface, False, True)
        surface = pg.transform.scale(surface, (self.width, self.height))
        self.screen.blit(surface, (0, 0))
        if self.draw_color is not None:
            for i in range(self.grid.dimensions[0]):
                pg.draw.line(self.screen, self.draw_color, (0, i * self.size_y), (self.width, i * self.size_y))
            for j in range(self.grid.dimensions[1]):
                pg.draw.line(self.screen, self.draw_color, (j * self.size_x, 0), (j * self.size_x, self.height))
        pg.display.update()
        return (time.time() - t0)

if __name__ == '__main__':
    pg.init()
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
    pattern = 'glider'
    resx, resy = 800, 800
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    if len(sys.argv) > 3:
        resx, resy = int(sys.argv[2]), int(sys.argv[3])
    if len(sys.argv) > 4:
        ITERATIONS = int(sys.argv[4])
    print(f"Pattern: {pattern}  Résolution: {resx}x{resy}  Itérations: {ITERATIONS}")
    try:
        init_pattern = dico_patterns[pattern]
    except KeyError:
        print("Pattern introuvable. Options :", list(dico_patterns.keys()))
        sys.exit(1)
    grid = Grille(*init_pattern)
    app = App((resx, resy), grid)
    
    total_calc = 0.0
    total_rendu = 0.0
    for it in range(ITERATIONS):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit(0)
        calc_time = grid.compute_next_iteration()
        total_calc += calc_time
        render_time = app.draw()
        total_rendu += render_time
        print(f"[It {it+1:03d}] Calcul: {calc_time:.4e}s, Rendu: {render_time:.4e}s")
    print(f"Total: Calcul: {total_calc:.4e}s, Rendu: {total_rendu:.4e}s sur {ITERATIONS} itérations")
    pg.quit()
