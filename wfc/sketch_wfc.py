import vsketch
from enum import Enum
import numpy as np
from itertools import compress
from collections import deque
from tqdm import trange

class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    
    def __str__(self):
        return self.name

# @njit(cache=True)
def dir_to_delta(dir):
    if dir == Direction.RIGHT:
        return np.array([0, 1])
    elif dir == Direction.DOWN:
        return np.array([1, 0])
    elif dir == Direction.LEFT:
        return np.array([0, -1])
    elif dir == Direction.UP:
        return np.array([-1, 0])

# @njit(cache=True)
def dir_to_cell(row, col, dir):
    delta = dir_to_delta(dir)
    return np.array([row, col]) + delta

def delta_to_dir(delta):
    if np.allclose(delta, np.array([0, 1])):
        return Direction.RIGHT
    elif np.allclose(delta, np.array([1, 0])):
        return Direction.DOWN
    elif np.allclose(delta, np.array([0, -1])):
        return Direction.LEFT
    elif np.allclose(delta, np.array([-1, 0])):
        return Direction.UP
    else:
        raise Exception(f"Invalid delta {delta}.")

def reverse_dir(dir):
    return Direction((dir.value + 2) % 4)

class Tile():
    def __init__(self, subsketch, prob, index) -> None:
        self.subsketch = subsketch
        self.prob = prob
        self.index = index
    
    def draw(self, vsk):
        vsk.sketch(self.subsketch)
        
    
class Rule():
    def __init__(self, tile, dir, other_tile, must_be) -> None:
        """_summary_

        Args:
            tile (_type_): _description_
            other_tile (_type_): _description_
            dir (_type_): _description_
            require (bool): If True, `tile` must be `dir` in relation to `other_tile`.
            If False, `tile` cannot be `dir` in relation to `other tile`.
        """
        self.tile = tile
        self.other_tile = other_tile
        self.dir = dir
        self.must_be = must_be
        
        
def draw_map(map, tileset, vsk, debug_indices=False, size=1.0):
    N_rows, N_cols = map.shape
    for row in range(N_rows):
        for col in range(N_cols):
            if debug_indices and int(map[row, col]) >= 0:
                vsk.stroke(2)
                vsk.vpype(f"text -f rowmans -s 8 -p {size * col + 0.05}cm {size * row + 0.2}cm \"T{int(map[row, col])}\"")
                vsk.stroke(1)
                
    with vsk.pushMatrix():
        vsk.translate(0.5, 0.5)
        for row in range(N_rows):
            with vsk.pushMatrix():
                for col in range(N_cols):
                    if int(map[row, col]) >= 0:
                        vsk.sketch(tileset[int(map[row, col])].subsketch)
                        vsk.translate(1.0, 0)
            vsk.translate(0, 1.0)
        

class WFC():
    def __init__(self, tileset, ruleset, N_rows, N_cols, invalid_edge_tiles=None) -> None:
        self.tileset = tileset
        self.ruleset = ruleset
        self.N_rows = N_rows
        self.N_cols = N_cols
        self.N_tiles = len(self.tileset)
        self.valid_edge_tiles = invalid_edge_tiles        
        
        self.possibilities = np.full((self.N_rows, self.N_cols, self.N_tiles), True)
        if invalid_edge_tiles is not None:  # remove invalid possibilties along edges
            for dir, tiles in invalid_edge_tiles.items():
                if dir == Direction.RIGHT:
                    self.possibilities[:,self.N_cols-1,tiles] = False
                elif dir == Direction.UP:
                    self.possibilities[0,:,tiles] = False
                elif dir == Direction.LEFT:
                    self.possibilities[:,0,tiles] = False
                elif dir == Direction.DOWN:
                    self.possibilities[self.N_rows-1,:,tiles] = False
        
        self.final_map = np.full((self.N_rows, self.N_cols), np.nan)
        self.entropy = self.calculate_entropy()
        
        self.neighbour_indices = np.array([[-1,0], [1,0], [0,-1], [0,1]])
    
    def draw(self, vsk, debug_indices=False, size=1.0):
        draw_map(self.final_map, self.tileset, vsk, debug_indices, size)
    
    def is_collapsed(self, row, col):
        return np.isnan(self.entropy[row, col])
    
    def get_valid_possibilities(self, row, col):
        return list(compress(self.tileset, self.possibilities[row, col]))
    
    def get_valid_possibilities_indices(self, row, col):
        return np.nonzero(self.possibilities[row, col])[0]
    
    def calculate_cell_entropy(self, row, col):
        valid_possibilities = self.get_valid_possibilities(row, col)
        p = np.array([possibility.prob for possibility in valid_possibilities])
        E = - np.sum(p * np.log(p))
        return E
    
    def calculate_entropy(self):
        entropy = np.zeros((self.N_rows, self.N_cols))
        for row in range(self.N_rows):
            for col in range(self.N_cols):
                entropy[row, col] = self.calculate_cell_entropy(row, col)
        return entropy
    
    def pick_lowest_entropy_cell(self, entropy):
        row, col = np.where(entropy == np.nanmin(entropy))
        idx = np.random.choice(row.shape[0])
        return row[idx], col[idx]
    
    def collapse_cell(self, row, col):
        valid_indices = np.where(self.possibilities[row, col])[0]
        probs = np.array([self.tileset[index].prob for index in valid_indices])
        probs = probs / np.sum(probs)
        picked_index = np.random.choice(valid_indices, p=probs)
        self.possibilities[row, col, :] = False
        self.possibilities[row, col, picked_index] = True
        self.final_map[row, col] = picked_index
        self.entropy[row, col] = np.nan
        return picked_index
    
    def neighbour_to_dir(self, row, col, row_neighbour, col_neighbour):
        delta = np.array([row_neighbour, col_neighbour]) - np.array([row, col])
        return delta_to_dir(delta)
    
    def get_valid_neighbours(self, row, col):
        """Retrieve valid neighbours of cell in (x,y).
        Valid means the neighbouring cells are within the bounds of the grid.
        """
        neighbour_indices = np.array([row, col]) + self.neighbour_indices
        neighbour_indices = neighbour_indices[(neighbour_indices[:,0] >= 0) & (neighbour_indices[:,0] < self.N_rows) & \
            (neighbour_indices[:,1] >= 0) & (neighbour_indices[:,1] < self.N_cols)]
        return neighbour_indices
    
    def get_valid_directions(self, row, col):
        neighbour_indices = self.get_valid_neighbours(row, col)
        dirs = []
        for row_n, col_n in neighbour_indices:
            dirs.append(self.neighbour_to_dir(row, col, row_n, col_n))
        return dirs
    
    def get_valid_noncollapsed_neighbours(self, row, col, entropy):
        """Retrieve valid noncollapsed neighbours of cell in (x,y).
        Valid means the neighbouring cells are 1. within the bounds of the grid and 2. not yet collapsed.
        """
        neighbour_indices = self.get_valid_neighbours(row, col)
        neighbour_indices = np.array([indices for indices in neighbour_indices if not np.isnan(entropy[indices[0],indices[1]])])
        return neighbour_indices
    
    def propagate(self, row_collapsed, col_collapsed, debug=False):
        # find neighbours that are not collapsed and not outside bounds
        # add them to FIFO queue
        # iterate
        # go through ruleset for the still valid choices
        # if any rules are not satisfied anymore, break and set possibility to False and continue to next possibility
        # therefore it would make sense for the ruleset to be categorized by tile_1 and tile_2. 
        # So when checking neighbour n for tile possibility t we need to check the rules of tile t that relates to the collapsed tile at x,y.
        # Need to think about this more. For now just check all.
        
        neighbour_indices = self.get_valid_noncollapsed_neighbours(row_collapsed, col_collapsed, self.entropy)
        np.random.shuffle(neighbour_indices)
        queue = deque([indices for indices in neighbour_indices])
        if debug: print("Neighbours in queue:\n", queue,"\n")
        
        while queue:
            row, col = queue.popleft()
            if debug: print(f"Popped ({row}, {col}).")
            
            valid_tile_indices = self.get_valid_possibilities_indices(row, col)
            if debug: print(f"Valid tiles: {valid_tile_indices}")
            
            valid_directions = self.get_valid_directions(row, col)
            # if debug: print(f"Valid directions: {[d.name for d in valid_directions]}")
            
            possibilties_updated = False
            
            for tile_index in valid_tile_indices:
                if debug: print(f"Checking tile {tile_index}")
                rules = self.ruleset[tile_index]
                
                rules_finished_forward = False
                for rule in rules:
                    # if debug: print(f"\tChecking rule: tile {rule.dir} of T{rule.tile.index} must be T{rule.other_tile.index} {rule.must_be}")
                    if rule.dir in valid_directions and not rules_finished_forward:
                        row_other, col_other = dir_to_cell(row, col, rule.dir)
                        # if debug: print(f"\t\tCell to check is ({row_other}, {col_other}).")
                        if (rule.must_be and not self.possibilities[row_other, col_other, rule.other_tile.index]) or \
                            (not rule.must_be and self.is_collapsed(row_other, col_other) and \
                            self.final_map[row_other, col_other] == rule.other_tile.index):
                                if debug: print(f"\t\t**Rule broken. Removing possibility {tile_index} from ({row}, {col}).**")
                                self.possibilities[row, col, tile_index] = False
                                possibilties_updated = True  # flag cell as updated
                                rules_finished_forward = True
                    
                    # reversed_dir = reverse_dir(rule.dir)
                    # if reversed_dir in valid_directions:
                    #     row_other, col_other = dir_to_cell(row, col, reversed_dir)
                    #     # if debug: print(f"\t\t(Reversed) cell to check is ({row_other}, {col_other}).")

                    #     if self.is_collapsed(row_other, col_other) and not rule.must_be and \
                    #         self.final_map[row_other, col_other] == rule.tile.index:
                    #         if debug: print(f"\t\t**(Reversed) rule broken. Removing possibility {rule.other_tile.index} from ({row}, {col}).**")
                    #         self.possibilities[row, col, rule.other_tile.index] = False
                    #         possibilties_updated = True  # flag cell as updated
                                
                    #     if self.is_collapsed(row_other, col_other) and rule.must_be and \
                    #         self.possibilities[row_other, col_other, rule.tile.index]:
                    #         if debug: print(f"\t\t**(Reversed) rule broken. Removing possibility {rule.other_tile.index} from ({row}, {col}).**")
                    #         self.possibilities[row, col, :] = False
                    #         self.possibilities[row, col, rule.other_tile.index] = True
                    #         possibilties_updated = True  # flag cell as updated
                            
            if possibilties_updated:
                self.entropy[row, col] = self.calculate_cell_entropy(row, col)  # update entropy
                if debug: print("Entropy:\n", self.entropy)
                valid_noncollapsed_neighbours = self.get_valid_noncollapsed_neighbours(row, col, self.entropy)
                np.random.shuffle(valid_noncollapsed_neighbours)
                if debug: print("Valid noncollapsed neighbours:\n", valid_noncollapsed_neighbours)

                for indices in valid_noncollapsed_neighbours:
                    if not any((indices == cell).all() for cell in queue):
                        queue.extend([indices])
                if debug: print("Updated queue:\n", queue)
            
            if debug: print("\n")
            
    def iterate(self, debug=False):
        row, col = self.pick_lowest_entropy_cell(self.entropy)
        tile_index = self.collapse_cell(row, col)
        if debug: print(f"Picked Tile {tile_index} at ({row}, {col}).")
        if debug: print("Map after collapse:\n", self.final_map)
        if debug: print("Entropy after collapse:\n", self.entropy)
        self.propagate(row, col, debug)
        if debug: print("Entropy after propagation:\n", self.entropy, "\n")
        return np.isnan(self.entropy).all()
    
    def solve(self, debug=False):
        # while not self.iterate(debug):
        #     pass
        for i in trange(self.N_rows * self.N_cols):
            self.iterate(debug)
            
class WfcSketch(vsketch.SketchClass):
    debug_grid = vsketch.Param(False)
    debug_indices = vsketch.Param(False)
    debug_tiles_order = vsketch.Param(False)
    debug_print = vsketch.Param(False)
    
    n_rows = vsketch.Param(10)
    n_cols = vsketch.Param(10)
    tile_size = vsketch.Param(1.0)
    width = vsketch.Param(0.1)
    radius = vsketch.Param(0.13)
    connected_edge = vsketch.Param(True)
    
    detail = "0.01mm"
    
    def draw_grid(self, vsk):
        vsk.stroke(2)
        for row in range(self.n_rows + 1):
            vsk.line(0, row, self.n_cols, row)
        for col in range(self.n_cols + 1):
            vsk.line(col, 0, col, self.n_rows)
        vsk.stroke(1)
        
    def new_tile_sketch(self):
        tile_sketch = vsketch.Vsketch()
        tile_sketch.detail(self.detail)
        return tile_sketch
        
    def generate_knot_tileset(self, width=0.1, radius=0.1):
        # probs = np.ones(17)
        # probs = np.ones(21)
        probs = np.array([1, 1, 1, 0.5, 0.5,
                          0.25, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25,
                          0.10, 0.10, 0.10, 0.10,
                          1, 0.5])
        probs = probs / np.sum(probs)
        
        tile_sketches = []
        for i in range(probs.shape[0]):
            tile_sketches.append(self.new_tile_sketch())
        
        # tile_sketches[0].rect(0, 0, 0.2, 0.2, mode="radius")
        
        # tile_sketches[1].circle(0, 0, 0.2, mode="radius")
        
        # tile_sketches[2].line(-0.2, 0, 0.2, 0.0)
        # tile_sketches[2].line(0, -0.2, 0.0, 0.2)
        
        for w in [width, -width]:
            tile_sketches[0].arc(-0.5, -0.5, 1.0 + w, 1.0 + w, 3*np.pi/2, 2*np.pi)
            tile_sketches[0].arc(0.5, 0.5, 1.0 + w, 1.0 + w, np.pi/2, np.pi)
        
            tile_sketches[1].arc(0.5,  -0.5, 1.0 + w, 1.0 + w, np.pi, 3*np.pi/2)
            tile_sketches[1].arc( -0.5, 0.5, 1.0 + w, 1.0 + w, 0, np.pi/2)
        
            tile_sketches[2].line(0.5 * w, -0.5, 0.5 * w, -0.5 * width)
            tile_sketches[2].line(0.5 * w, 0.5 * width, 0.5 * w, 0.5)
            # tile_sketches[2].line(-0.5, 0.5 * w, 0.5, 0.5 * w)
            tile_sketches[2].line(-0.5, 0.5 * w, -0.5 * width, 0.5 * w)
            tile_sketches[2].line(0.5, 0.5 * w, 0.5 * width, 0.5 * w)
            
            tile_sketches[3].line(0.5 * w, -0.5, 0.5 * w, 0.5)
            tile_sketches[4].line(-0.5, 0.5 * w, 0.5, 0.5 * w)
            
            tile_sketches[5].arc(-0.5, -0.5, 1.0 + w, 1.0 + w, 3*np.pi/2, 2*np.pi)

            tile_sketches[9].line(0.5 * w, -0.5, 0.5 * w, -0.5 * width)
            
            # tile_sketches[17].line(0, -0.5, 0, 0)
            
            length = radius * np.sin(np.arccos(0.5 * width / radius))
            tile_sketches[17].line(0.5 * w, -0.5, 0.5 * w, -length)
            tile_sketches[17].line(0.5 * w, length, 0.5 * w, 0.5)
            tile_sketches[17].line(-0.5, 0.5 * w, -length, 0.5 * w)
            tile_sketches[17].line(0.5, 0.5 * w, length, 0.5 * w)
            tile_sketches[17].circle(0, 0, radius=radius)
            
            tile_sketches[18].sketch(tile_sketches[17])
            tile_sketches[18].circle(0, 0, 1e-4)
            
        
        tile_sketches[9].line(-0.5, -0.5 * width, -0.5 * width, -0.5 * width)
        tile_sketches[9].line(0.5 * width   , -0.5 * width, 0.5, -0.5 * width)
        tile_sketches[9].line(-0.5, 0.5 * width, 0.5, 0.5 * width)
        # tile_sketches[9].circle(0, 0, radius=radius)
        
        tile_sketches[13].line(-0.5 * width, -0.5, -0.5 * width, 0.5 * width)
        tile_sketches[13].line(0.5 * width, -0.5, 0.5 * width, -0.5 * width)
        tile_sketches[13].line(0.5 * width, -0.5 * width, 0.5, -0.5 * width)
        tile_sketches[13].line(-0.5 * width, 0.5 * width, 0.5, 0.5 * width)
        
        for i in [6, 7, 8, 10, 11, 12, 14, 15, 16]:
            tile_sketches[i].rotate(0.5 * np.pi)
            tile_sketches[i].sketch(tile_sketches[i-1])
        
        tiles = []
        for index, (sketch, prob) in enumerate(zip(tile_sketches, probs)):
            tiles.append(Tile(sketch, prob, index))
        return tiles
            
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.tile_size, self.tile_size)
        
        if self.debug_grid:
            self.draw_grid(vsk)
                    
        tileset = self.generate_knot_tileset(width=self.width, radius=self.radius)
        
        n_tiles = len(tileset)
        
        ruleset = [[] for i in range(n_tiles)]
        valid_directions = [[Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN],  # 0  
                            [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN],  # 1
                            [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN],  # 2
                            [                 Direction.UP,                 Direction.DOWN],  # 3
                            [Direction.RIGHT,               Direction.LEFT,               ],  # 4
                            [                 Direction.UP, Direction.LEFT                ],  # 5
                            [Direction.RIGHT, Direction.UP                                ],  # 6
                            [Direction.RIGHT,                               Direction.DOWN],  # 7
                            [                               Direction.LEFT, Direction.DOWN],  # 8
                            [Direction.RIGHT, Direction.UP, Direction.LEFT                ],  # 9
                            [Direction.RIGHT, Direction.UP                , Direction.DOWN],  # 10
                            [Direction.RIGHT,               Direction.LEFT, Direction.DOWN],  # 11
                            [                 Direction.UP, Direction.LEFT, Direction.DOWN],  # 12
                            # [Direction.UP],                                                 # 13
                            # [Direction.RIGHT],                                              # 14
                            # [Direction.DOWN],                                               # 15
                            # [Direction.LEFT],                                               # 16
                            [Direction.RIGHT, Direction.UP                                ],  # 13
                            [Direction.RIGHT,                               Direction.DOWN],  # 14
                            [                               Direction.LEFT, Direction.DOWN],  # 15
                            [                 Direction.UP, Direction.LEFT                ],  # 16
                            [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN],  # 17
                            [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN]]  # 18
        
        # Generate all rules based on valid_directions:
        for i, valid_dirs in enumerate(valid_directions):
            for j, other_valid_dirs in enumerate(valid_directions):
                for dir in Direction:
                    opposite_dir = reverse_dir(dir)
                    if (dir in valid_dirs and opposite_dir not in other_valid_dirs) or \
                        (dir not in valid_dirs and opposite_dir in other_valid_dirs):
                            ruleset[i].append(Rule(tileset[i], dir, tileset[j], must_be=False))
        
        # Generate valid tiles on edges:
        if self.connected_edge:
            invalid_edge_tiles = {Direction.RIGHT: [], Direction.UP: [], Direction.LEFT: [], Direction.DOWN: []}
            for i, directions in enumerate(valid_directions):
                for dir in Direction:
                    if dir in directions:
                        invalid_edge_tiles[dir].append(i)
        else:
            invalid_edge_tiles = None
            
        if self.debug_tiles_order:
            map = np.arange(self.n_rows * self.n_cols).reshape((self.n_rows, self.n_cols))
            map[map >= n_tiles] = -1
            draw_map(map, tileset, vsk, debug_indices=self.debug_indices, size=self.tile_size)
        else:
            wfc = WFC(tileset, ruleset, self.n_rows, self.n_cols, invalid_edge_tiles)
            wfc.solve(debug=self.debug_print)
            wfc.draw(vsk, debug_indices=self.debug_indices, size=self.tile_size)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    WfcSketch.display()
