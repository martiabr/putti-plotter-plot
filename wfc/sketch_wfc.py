import vsketch
from enum import Enum
import numpy as np
from itertools import compress
from collections import deque
from tqdm import trange, tqdm


class Direction(Enum):
    # RIGHT = 0
    # UP = 1
    # LEFT = 2
    # DOWN = 3
    RIGHT = 0
    UPPER_RIGHT = 1
    UP = 2
    UPPER_LEFT = 3
    LEFT = 4
    LOWER_LEFT = 5
    DOWN = 6
    LOWER_RIGHT = 7
    
    def __str__(self):
        return self.name


def dir_to_delta(dir):
    # if dir == Direction.RIGHT:
    #     return np.array([0, 1])
    # elif dir == Direction.DOWN:
    #     return np.array([1, 0])
    # elif dir == Direction.LEFT:
    #     return np.array([0, -1])
    # elif dir == Direction.UP:
    #     return np.array([-1, 0])
    if dir == Direction.RIGHT:
        return np.array([0, 1])
    elif dir == Direction.UPPER_RIGHT:
        return np.array([-1, 1])
    elif dir == Direction.UP:
        return np.array([-1, 0])
    elif dir == Direction.UPPER_LEFT:
        return np.array([-1, -1])
    elif dir == Direction.LEFT:
        return np.array([0, -1])
    elif dir == Direction.LOWER_LEFT:
        return np.array([1, -1])
    elif dir == Direction.DOWN:
        return np.array([1, 0])
    elif dir == Direction.LOWER_RIGHT:
        return np.array([1, 1])


def dir_to_cell(row, col, dir):
    delta = dir_to_delta(dir)
    return np.array([row, col], dtype=int) + delta


def delta_to_dir(delta):
    # if np.allclose(delta, np.array([0, 1])):
    #     return Direction.RIGHT
    # elif np.allclose(delta, np.array([1, 0])):
    #     return Direction.DOWN
    # elif np.allclose(delta, np.array([0, -1])):
    #     return Direction.LEFT
    # elif np.allclose(delta, np.array([-1, 0])):
    #     return Direction.UP
    if np.allclose(delta, np.array([0, 1])):
        return Direction.RIGHT
    elif np.allclose(delta, np.array([-1, 1])):
        return Direction.UPPER_RIGHT
    elif np.allclose(delta, np.array([-1, 0])):
        return Direction.UP
    elif np.allclose(delta, np.array([-1, -1])):
        return Direction.UPPER_LEFT
    elif np.allclose(delta, np.array([0, -1])):
        return Direction.LEFT
    elif np.allclose(delta, np.array([1, -1])):
        return Direction.LOWER_LEFT
    elif np.allclose(delta, np.array([1, 0])):
        return Direction.DOWN
    elif np.allclose(delta, np.array([1, 1])):
        return Direction.LOWER_RIGHT
    else:
        raise Exception(f"Invalid delta {delta}.")


def reverse_dir(dir):
    return Direction((dir.value + 4) % 8)


class Tile():
    def __init__(self, index, prob, valid_dirs, sketches) -> None: 
        self.index = index
        self.prob = prob
        self.sketches = sketches
        self.valid_dirs = valid_dirs
    
    def draw(self, vsk, layers=None):
        if layers is None:
            for sketch in self.sketches.values():
                vsk.sketch(sketch)
        else:
            for key, sketch in self.sketches.items():
                vsk.stroke(layers[key] + 1)  # note: +1 just to avoid stroke 0
                vsk.sketch(sketch)
            vsk.stroke(1)
        
    
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
    
    def __str__(self):
        if self.must_be:
            return f"Rule T{self.tile.index} must be {self.dir} of T{self.other_tile.index}"
        else:
            return f"Rule T{self.tile.index} cannot be {self.dir} of T{self.other_tile.index}"
        
        
def draw_map(map, tileset, vsk, debug_tile_indices=False, debug_cell_indices=False, possibilities=None, 
             valid_dirs=None, layers=None, size=1.0):
    N_rows, N_cols = map.shape
    for row in range(N_rows):
        for col in range(N_cols):
                debug_str = ""
                if debug_cell_indices:
                    debug_str += f"({row},{col}): "
                if debug_tile_indices:
                    if np.isnan(map[row, col]):
                        if possibilities is not None:
                            debug_str += f"P{int(np.sum(possibilities[row, col]))}"
                        else:
                            debug_str += "X"
                    else:
                        debug_str += f"T{int(map[row, col])}"
                vsk.stroke(2)
                vsk.vpype(f"text -f rowmans -s 6 -p {size * col + 0.05}cm {size * row + 0.2}cm \"{debug_str}\"")
                vsk.stroke(1)
                
    with vsk.pushMatrix():
        vsk.translate(0.5, 0.5)
        for row in range(N_rows):
            with vsk.pushMatrix():
                for col in range(N_cols):
                    if not np.isnan(map[row, col]):
                        tile = int(map[row, col])
                        if tile >= 0:
                            if layers is not None:
                                tileset[tile].draw(vsk, layers[row][col])
                            else:
                                tileset[tile].draw(vsk)
                            
                            if valid_dirs is not None:
                                vsk.stroke(2)
                                for dir in valid_dirs[tile]:
                                    vsk.circle(0.4 * dir_to_delta(dir)[1], 0.4 * dir_to_delta(dir)[0], 0.05 * size)
                                # for key in tileset[tile].sketches.keys():
                                #     if key in [dir for dir in Direction]:
                                #         vsk.circle(0.4 * dir_to_delta(key)[1], 0.4 * dir_to_delta(key)[0], 0.05 * size)
                                    
                                vsk.stroke(1)
                            
                    vsk.translate(1.0, 0)
            vsk.translate(0, 1.0)
        

class WFC():
    def __init__(self, tileset, ruleset, N_rows, N_cols, init_map=None, invalid_edge_tiles=None, debug=False) -> None:
        self.tileset = tileset
        self.ruleset = ruleset
        self.N_rows = N_rows
        self.N_cols = N_cols
        self.N_tiles = len(self.tileset)
        self.valid_edge_tiles = invalid_edge_tiles  
        self.debug = debug
        
        # self.neighbour_indices = np.array([[-1,0], [1,0], [0,-1], [0,1]])
        self.neighbour_indices = np.array([[0,1], [-1,1], [-1,0], [-1,-1], [0,-1], [1,-1], [1,0], [1,1]])
        
        self.possibilities = np.full((self.N_rows, self.N_cols, self.N_tiles), True)
        if invalid_edge_tiles is not None:  # remove invalid possibilties along edges
            self.remove_edge_possibilities(invalid_edge_tiles)
        self.init_possibilities = np.copy(self.possibilities)
            
        self.entropy = self.calculate_entropy()
            
        if init_map is not None:
            self.final_map = init_map.copy()
            self.collapsed = ~np.isnan(init_map)
            init_stack = []
            for r, c in np.argwhere(self.collapsed):
                # Collapse probabilities:
                self.possibilities[r, c, :] = False
                self.possibilities[r, c, int(init_map[r, c])] = True 

                # Add neighbours to stack to update neighbouring probabilities:
                neighbour_indices = self.get_valid_noncollapsed_neighbours(r, c)
                neighbour_indices = [(i, j) for i, j in neighbour_indices if (i, j) not in init_stack]
                init_stack.extend(neighbour_indices)
            self.propagate_constraints(init_stack, repair=False)
        else:
            self.final_map = np.full((self.N_rows, self.N_cols), np.nan)
            self.collapsed = np.full((self.N_rows, self.N_cols), False)
        self.entropy = self.calculate_entropy()
                    
    def remove_edge_possibilities(self, invalid_edge_tiles):
        for dir, tiles in invalid_edge_tiles.items():
            if dir == Direction.RIGHT:
                self.possibilities[:,self.N_cols-1,tiles] = False
            elif dir == Direction.UP:
                self.possibilities[0,:,tiles] = False
            elif dir == Direction.LEFT:
                self.possibilities[:,0,tiles] = False
            elif dir == Direction.DOWN:
                self.possibilities[self.N_rows-1,:,tiles] = False
                
    def draw(self, vsk, debug_tile_indices=False, debug_cell_indices=False, valid_dirs=None, layers=None, size=1.0):
        draw_map(self.final_map, self.tileset, vsk, debug_tile_indices=debug_tile_indices, 
                 debug_cell_indices=debug_cell_indices, valid_dirs=valid_dirs, possibilities=self.possibilities, 
                 layers=layers, size=size)
    
    def is_collapsed(self, row, col):
        return self.collapsed[row, col]
    
    def get_valid_possibilities(self, row, col):
        return list(compress(self.tileset, self.possibilities[row, col]))
    
    def get_valid_possibilities_indices(self, row, col):
        return np.nonzero(self.possibilities[row, col])[0]
    
    def calculate_cell_entropy(self, row, col):
        valid_possibilities = self.get_valid_possibilities(row, col)
        p = np.array([possibility.prob for possibility in valid_possibilities])
        E = - np.sum(p * np.log(p))
        # E = np.sum(p)
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
    
    def collapse_cell(self, row, col, fallback_index=0):
        valid_indices = np.where(self.possibilities[row, col])[0]
        if len(valid_indices) > 0:
            probs = np.array([self.tileset[index].prob for index in valid_indices])
            probs = probs / np.sum(probs)
            picked_index = np.random.choice(valid_indices, p=probs)
        else:
            print(f"No valid tiles for ({row}, {col}).")
            picked_index = fallback_index
        self.possibilities[row, col, :] = False
        self.possibilities[row, col, picked_index] = True
        self.final_map[row, col] = picked_index
        self.collapsed[row, col] = True
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
    
    def get_valid_noncollapsed_neighbours(self, row, col):
        """Retrieve valid noncollapsed neighbours of cell in (x,y).
        Valid means the neighbouring cells are 1. within the bounds of the grid and 2. not yet collapsed.
        """
        neighbour_indices = self.get_valid_neighbours(row, col)
        # neighbour_indices = np.array([indices for indices in neighbour_indices if not np.isnan(entropy[indices[0],indices[1]])])
        neighbour_indices = np.array([indices for indices in neighbour_indices if not self.collapsed[indices[0],indices[1]]])
        return neighbour_indices
    
    def propagate_constraints(self, stack, repair=False):    
        stack = stack[:]  # copy
        in_stack = set(stack)
        # if self.debug: print("Initial neighbours in stack:", stack)
        
        cells_updated = set()
        
        while stack:
            # row, col = queue.popleft()
            row, col = stack.pop()
            in_stack.remove((row, col))
            # if self.debug: print(f"Popped ({row}, {col}).")
            
            valid_tile_indices = self.get_valid_possibilities_indices(row, col)
            valid_directions = self.get_valid_directions(row, col)
            possibilties_updated = False
            # if self.debug: print(f"Valid tiles: {valid_tile_indices}")
            # if self.debug: print(f"Valid directions: {[d.name for d in valid_directions]}")
            
            for tile_index in valid_tile_indices:
                # if self.debug: print(f"Checking tile {tile_index}")
                rules = self.ruleset[tile_index]
                for rule in rules:
                    # if self.debug: print(f"\tChecking rule: tile {rule.dir} of T{rule.tile.index} must be T{rule.other_tile.index} {rule.must_be}")
                    if rule.dir in valid_directions:
                        row_other, col_other = dir_to_cell(row, col, rule.dir)
                        # if self.debug: print(f"\t\tCell to check is ({row_other}, {col_other}).")
                        if (rule.must_be and not self.possibilities[row_other, col_other, rule.other_tile.index]) or \
                            (not rule.must_be and self.is_collapsed(row_other, col_other) and \
                            self.final_map[row_other, col_other] == rule.other_tile.index):
                                if self.debug: print(f"\t**'{rule}' broken. Removing possibility T{tile_index} from ({row}, {col}).**")
                                # print(row_other, col_other, rule.other_tile.index)
                                # print(f"({row}, {col}) {rule}")
                                # print(self.is_collapsed(row_other, col_other), self.final_map[row_other, col_other])
                                self.possibilities[row, col, tile_index] = False
                                possibilties_updated = True  # flag cell as updated
                                break
                            
            if possibilties_updated:
                cells_updated.add((row, col))
                
                # Add cell and all 8 neighbours to stack and reset probabilities if stuck:
                if repair and np.sum(self.possibilities[row, col]) == 0:
                    if True: print(f"No possibilities for ({row}, {col}). Attempting to repair.")
                    cells_to_reset = self.get_valid_neighbours(row, col)
                    np.random.shuffle(cells_to_reset)
                    cells_to_reset = [[row, col]] + cells_to_reset.tolist()
                    for r, c in cells_to_reset:  # TODO: dont do loop
                        self.possibilities[r, c] = self.init_possibilities[r, c]
                        self.collapsed[r, c] = False
                        self.final_map[r, c] = np.nan
                        if (r, c) not in in_stack:
                            stack.append((r, c))
                            in_stack.add((r, c))
                    if self.debug: print(f"Repair applied. Reset and added {len(cells_to_reset)} cells back to the stack.")
                else:
                    valid_noncollapsed_neighbours = self.get_valid_noncollapsed_neighbours(row, col)
                    # np.random.shuffle(valid_noncollapsed_neighbours)
                    # if self.debug: print("Valid noncollapsed neighbours:\n", valid_noncollapsed_neighbours)

                    # Add neighbours to stack:
                    for indices in valid_noncollapsed_neighbours:
                        neighbor_tuple = tuple(indices)
                        if neighbor_tuple not in in_stack:
                            stack.append(neighbor_tuple)
                            in_stack.add(neighbor_tuple)
                    # if self.debug: print("Updated stack:\n", stack)
            
            # if self.debug: print("\n")
            
        # Update entropy for cells that were updated:
        for row, col in cells_updated:  
            self.entropy[row, col] = self.calculate_cell_entropy(row, col)
            # if self.debug: print("Entropy:\n", self.entropy)
    
    def propagate_constraints_from(self, row, col, repair=False):
        neighbour_indices = self.get_valid_noncollapsed_neighbours(row, col)
        if neighbour_indices.shape[0] > 0:
            # np.random.shuffle(neighbour_indices)
            stack = [tuple(indices) for indices in neighbour_indices]
            self.propagate_constraints(stack, repair=repair)
         
    def iterate(self):
        # print(self.collapsed)
        row, col = self.pick_lowest_entropy_cell(self.entropy)
        tile_index = self.collapse_cell(row, col)
        if self.debug: 
            print(f"Picked Tile {tile_index} at ({row}, {col}).")
            print("Map after collapse:\n", self.final_map)
            # print("Entropy after collapse:\n", self.entropy)
        self.propagate_constraints_from(row, col, repair=True)
        if self.debug: 
            # print("Entropy after propagation:\n", self.entropy, "\n")
            print("Map after propagation:\n", self.final_map)
        return self.collapsed.all()
    
    def solve(self):
        with tqdm(total=self.N_rows * self.N_cols) as pbar:
            while not self.iterate():
                pbar.update(1)
            pbar.update(1)
        
        # for i in trange(self.N_rows * self.N_cols):
        # for i in trange(49):
        #     self.iterate()

        self.map_layers = self.solve_layers(self.final_map)
        
    def solve_layers(self, tile_map):
        directions = [dir for dir in Direction]
        visited = np.full((self.N_rows, self.N_cols), False)
        stack = deque([[(row, col), (None, None)] for row in range(self.N_rows) for col in range(self.N_cols)])
        
        layer_count = 1
        layers_all = [[{} for _ in range(self.N_cols)] for _ in range(self.N_rows)]  # for each key in (row, col) assign a layer int
        
        # Search algo:
        # If has prev: get dir from prev to curr, get layer on prev (in dir if multiple)
        # Otherwise assign new colour

        # if single sketch, just assign layer, otherwise set layer in reverse dir, non-dir layers are set to default (0)
        
        # if two dirs: get next dir, get next cell, append
        # if more: 
            # if has forward dir, colour it and append (after any other so it is popped first - depth first)
            # other dirs are given new colours and apoended (before forward)
            # if opposite they are given same colour/layer
        
        # Choose either to have loop of each tile + stack inside loop, or just a stack. If so, how to keep track of previous tile etc...
        # for row, col in [(r, c) for r in range(self.N_rows) for c in range(self.N_cols)]:
            # if not visited[row, col]:
            #     stack = deque([[(row, col), (None, None)]])  # [(row, col), (from_row, from_col)]
            #     while stack:
            
        while stack:
            (row, col), (r_prev, c_prev) = stack.pop()
            if not visited[row, col]:
                tile_index = int(tile_map[row, col])
                tile = self.tileset[tile_index]
                layers = list(tile.sketches.keys())
                
                # valid_dirs = self.get_valid_directions(row, col)  # TODO: this is wrong
                # valid_dirs = tile.valid_dirs  # TODO: need to check for border
                valid_dirs = []
                for dir in tile.valid_dirs:
                    cell = dir_to_cell(row, col, dir)
                    if cell[0] >= 0 and cell[1] >= 0 and cell[0] < self.N_rows and cell[1] < self.N_cols:
                        valid_dirs.append(dir)
                
                # print(f"stack: {stack} ")
                print(f"*** Popped ({row}, {col}): {layers}, valid_dirs: {valid_dirs}, prev: ({r_prev}, {c_prev})")
                
                # Colour assignment:
                if r_prev is not None:  # if there is a prev tile
                    # Get prev tile's layers:
                    prev_dir = delta_to_dir(np.array([row, col]) - np.array([r_prev, c_prev]))
                    prev_dir_reverse = reverse_dir(prev_dir)
                    prev_tile_index = int(tile_map[r_prev, c_prev])
                    prev_tile = self.tileset[prev_tile_index]
                    prev_layers = list(prev_tile.sketches.keys())
                    
                    # Get color from prev tile to propagate to curr tile:
                    prev_color = 0
                    if len(prev_layers) == 1:
                        layer = prev_layers[0]
                        prev_color = layers_all[r_prev][c_prev][layer]
                    elif prev_dir in prev_layers:
                        prev_color = layers_all[r_prev][c_prev][prev_dir]
                    # print("prev_color", prev_color)
                    # print("prev_dir", prev_dir)
                    
                    # Assign color from prev tile to curr tile:
                    if len(layers) == 1:
                        layers_all[row][col][layers[0]] = prev_color
                    else:
                        if prev_dir_reverse in layers:  # reverse colour should be colored same
                            layers_all[row][col][prev_dir_reverse] = prev_color
                        if prev_dir in layers:  # forward colour should be colored same
                            layers_all[row][col][prev_dir] = prev_color
                        
                        for layer in layers:
                            if layer not in directions:  # not dir layers we can just set to default (zero)
                                layers_all[row][col][layer] = 0
                        # for layer in layers:  # assign remaining layers as new colours (except ones not a direction which go to default (0))
                        #     if layer not in layers_all[row][col].keys():
                        #         if layer in directions:
                        #             layers_all[row][col][layer] = layer_count
                        #             layer_count += 1
                        #             # TODO: we just want one new colour for the rest in directions, not one new for each
                        #         else:
                        #             layers_all[row][col][layer] = 0
                    print("colored:", layers_all[row][col])
                    # TODO: refactor to avoid big outer if else?
                    
                    # Append new cells:
                    # for dir in valid_dirs:
                    #     if dir not in (prev_dir, prev_dir_reverse):  # add all cells except reverse (thats where we came from), and forward (important to add this on top of stack after others)
                    #         r_n, c_n = dir_to_cell(row, col, dir)
                    #         stack.append(((r_n, c_n), (row, col)))
                    #         # print("appended not forward cell:", r_n, c_n)
                    if len(valid_dirs) == 2:  # if only 2 dirs, just add not where we came from
                        for dir in valid_dirs:
                            if dir != prev_dir_reverse:
                                r_n, c_n = dir_to_cell(row, col, dir)
                                stack.append(((r_n, c_n), (row, col)))
                                print("appended forward:", (r_n, c_n))
                    elif len(valid_dirs) > 2:  # if more than 2 dirs, go forward if it exists (the other dirs we handle later since we dont yet know their color)
                        if prev_dir in valid_dirs:
                            r_n, c_n = dir_to_cell(row, col, prev_dir)
                            stack.append(((r_n, c_n), (row, col)))
                            print("appended prev_dir:", (r_n, c_n))
                            
                    # if prev_dir in valid_dirs:  # 
                    #     r_n, c_n = dir_to_cell(row, col, prev_dir)
                    #     stack.append(((r_n, c_n), (row, col)))
                    #     # print("appended cell with prev_dir:", r_n, c_n)
                
                else:  # if there is no prev tile
                    if len(valid_dirs) > 0:
                        if len(valid_dirs) in (1, 2):  # if single or two dirs, go in all dirs
                            dirs = valid_dirs
                        else:  # if more than two directions, pick random and leave the rest
                            dirs = [np.random.choice(valid_dirs)]
                            if reverse_dir(dirs[0]) in valid_dirs:  # but if reversed dir is there, add that as well
                                dirs.append(reverse_dir(dirs[0]))
                            # TODO: make sure cell is removed from stack
                        
                        if len(layers) == 1:  # if single layer just assign new color
                            layers_all[row][col][layers[0]] = layer_count
                            layer_count += 1
                        else:
                            for layer in layers:
                                if layer in dirs:  # colour layers we are moving in
                                    layers_all[row][col][layer] = layer_count
                                else:
                                    layers_all[row][col][layer] = 0
                            layer_count += 1  # if moving in reverse dir as well, they should be same color so only increment after
                         
                        print("colored:", layers_all[row][col])
                        
                        # for layer in layers:
                        #     if layer in directions or len(layers) == 1:  # if single layer, assign new color, otherwise assign all dir layers to new colors
                        #         layers_all[row][col][layer] = layer_count
                        #         layer_count += 1
                        #     else:
                        #         layers_all[row][col][layer] = 0
                            
                        for dir in dirs:
                            r_n, c_n = dir_to_cell(row, col, dir)
                            stack.append(((r_n, c_n), (row, col)))
                            print("appended", r_n, c_n)
                        
                    # # Append all cells, order is irrelevant when not coming from somewhere else:
                    # for dir in valid_dirs:
                    #     # Add neighbour to stack:
                    #     r_n, c_n = dir_to_cell(row, col, dir)
                    #     stack.append(((r_n, c_n), (row, col)))
                    #     # TODO: refactor away this operation to avoid copy paste
                    #     # print("appended", r_n, c_n)
                
                # when popping, need to assign color 
                # if didnt come from somewhere, set all new colours
                # if came somewhere, that part is that colour, handle main, and set others to new colours
                # in case of just one layer then all is that colour
                
                # for each item in stack, need to keep track of 
                # 1. where we came from
                # 2. 
                
                # Set visited to true if all (dir) layers have been set:
                if all(k in layers_all[row][col].keys() for k in layers):
                    visited[row, col] = True
                    print(f"Visit {row}, {col} completed")
                # visited[row, col] = True
        
        return layers_all
        
        # specify tiles to start iterating from
        # need a default layer for cells we dont visit - could extend algo to guarantee that we visit all cells
        # tiles now have a main sketch + one sketch per direction amongst its valid directions
        # if only one other direction, add next cell to same layer
        # depth first
        # if other directions, they are added to stack after the main forward direction
        
        # alternative is simply to iterate through all cells in order.
        # when we hit a tile with other directions start depth first searching
        # keeping track of each isolated track
        # should work fine, just remember to add dir if 1 dir, both dirs of same layer if back/forwards, 
        # two different layer dirs if two not back/forwards, and so on.
        # since the depth first search searches anything it can, there should not be a case where multiple different colors meet head on.
        
        # To start searching we need:
            # separate tile sketch into dict with keys being directions + main. kind of messy. Could separate main and otehr dirs...
            # also need to store layer/color of each of these sketches - this is only needed in solve_layers.
            # So it can be separated from the tile class?
            
        
        # Another question though is merging layers that do not connect. both what algo to use and how to actually go through the tiles and merge.
        # Could do an actual graph colouring problem but that seems overkill. How would it look?
            # Each layer (corresponding to single line) is a node
            # Edges are connections between the lines
            # The easiest is simply to construct this graph while running the search that assigns layers to the tiles
            # All of this also mean it would be nasty to set the layers in the main data structure, rather use something intermediate, and set at the end after graph colouring
            # A problem here is that we dont necessarily want the minimum number of colours.
            # Greedy colouring: assign first available colour not in use by any neighbour
            # A result of this is that layer 0, 1, 2 is used a lot while upper colours are not used
            # An augmented version of this is "we have n colours, use a random colour that is not in use by any neighbour"
            # Or "load-balancing" my keeping track of count of each layer and choosing the least used layer. 
            # Or include some patial information.




        
          
class WfcSketch(vsketch.SketchClass):
    debug_grid = vsketch.Param(False)
    debug_tile_indices = vsketch.Param(False)
    debug_cell_indices = vsketch.Param(False)
    debug_valid_dirs = vsketch.Param(False)
    debug_tiles_order = vsketch.Param(False)
    debug_print = vsketch.Param(False)
    
    n_rows = vsketch.Param(10)
    n_cols = vsketch.Param(10)
    tile_size = vsketch.Param(1.0)
    width = vsketch.Param(0.1)
    radius_circles = vsketch.Param(0.15)
    radius_turns = vsketch.Param(0.5)
    connected_edge = vsketch.Param(True)
    tileset = vsketch.Param("knots", choices=["knots", "metro"])
    use_custom_init = vsketch.Param(False)
    
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
        probs = np.array([1, 1, 1, 0.5, 0.5,
                          0.25, 0.25, 0.25, 0.25,
                          0.25, 0.25, 0.25, 0.25,
                          0.10, 0.10, 0.10, 0.10,
                          1, 0.5])
        probs = probs / np.sum(probs)
        
        tile_sketches = []
        for i in range(probs.shape[0]):
            tile_sketches.append({"main": self.new_tile_sketch()})
        
        for w in [width, -width]:
            tile_sketches[0]["main"].arc(-0.5, -0.5, 1.0 + w, 1.0 + w, 3*np.pi/2, 2*np.pi)
            tile_sketches[0]["main"].arc(0.5, 0.5, 1.0 + w, 1.0 + w, np.pi/2, np.pi)
        
            tile_sketches[1]["main"].arc(0.5,  -0.5, 1.0 + w, 1.0 + w, np.pi, 3*np.pi/2)
            tile_sketches[1]["main"].arc( -0.5, 0.5, 1.0 + w, 1.0 + w, 0, np.pi/2)
        
            tile_sketches[2]["main"].line(0.5 * w, -0.5, 0.5 * w, -0.5 * width)
            tile_sketches[2]["main"].line(0.5 * w, 0.5 * width, 0.5 * w, 0.5)
            tile_sketches[2]["main"].line(-0.5, 0.5 * w, -0.5 * width, 0.5 * w)
            tile_sketches[2]["main"].line(0.5, 0.5 * w, 0.5 * width, 0.5 * w)
            
            tile_sketches[3]["main"].line(0.5 * w, -0.5, 0.5 * w, 0.5)
            tile_sketches[4]["main"].line(-0.5, 0.5 * w, 0.5, 0.5 * w)
            
            tile_sketches[5]["main"].arc(-0.5, -0.5, 1.0 + w, 1.0 + w, 3*np.pi/2, 2*np.pi)

            tile_sketches[9]["main"].line(0.5 * w, -0.5, 0.5 * w, -0.5 * width)
            
            length = radius * np.sin(np.arccos(0.5 * width / radius))
            tile_sketches[17]["main"].line(0.5 * w, -0.5, 0.5 * w, -length)
            tile_sketches[17]["main"].line(0.5 * w, length, 0.5 * w, 0.5)
            tile_sketches[17]["main"].line(-0.5, 0.5 * w, -length, 0.5 * w)
            tile_sketches[17]["main"].line(0.5, 0.5 * w, length, 0.5 * w)
            tile_sketches[17]["main"].circle(0, 0, radius=radius)
            
            tile_sketches[18]["main"].sketch(tile_sketches[17]["main"])
            tile_sketches[18]["main"].circle(0, 0, 1e-4)
            
        
        tile_sketches[9]["main"].line(-0.5, -0.5 * width, -0.5 * width, -0.5 * width)
        tile_sketches[9]["main"].line(0.5 * width   , -0.5 * width, 0.5, -0.5 * width)
        tile_sketches[9]["main"].line(-0.5, 0.5 * width, 0.5, 0.5 * width)
        
        tile_sketches[13]["main"].line(-0.5 * width, -0.5, -0.5 * width, 0.5 * width)
        tile_sketches[13]["main"].line(0.5 * width, -0.5, 0.5 * width, -0.5 * width)
        tile_sketches[13]["main"].line(0.5 * width, -0.5 * width, 0.5, -0.5 * width)
        tile_sketches[13]["main"].line(-0.5 * width, 0.5 * width, 0.5, 0.5 * width)
        
        for i in [6, 7, 8, 10, 11, 12, 14, 15, 16]:
            tile_sketches[i]["main"].rotate(0.5 * np.pi)
            tile_sketches[i]["main"].sketch(tile_sketches[i-1]["main"])
        
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
        
        tiles = []
        for index, (prob, valid_dirs, sketches) in enumerate(zip(probs, valid_directions, tile_sketches)):
            tiles.append(Tile(index, prob, valid_dirs, sketches)) 
            
        # Generate all rules based on valid_directions:
        ruleset = [[] for i in range(len(tiles))]
        for i, valid_dirs in enumerate(valid_directions):
            for j, other_valid_dirs in enumerate(valid_directions):
                for dir in Direction:
                    opposite_dir = reverse_dir(dir)
                    # "If tile i connects to the right and tile j does not connect to the left, the relation is illegal"
                    if (dir in valid_dirs and opposite_dir not in other_valid_dirs) or \
                        (dir not in valid_dirs and opposite_dir in other_valid_dirs):
                            ruleset[i].append(Rule(tiles[i], dir, tiles[j], must_be=False))
        
        return tiles, valid_directions, ruleset

    def generate_metro_tileset(self, circle_radius=0.1, turn_radius=0.5, stop_length=0.5):
        probs_list = [[10], 2*[3], 2*[3], 
                      4*[1], 4*[1], 4*[1], # turns
                      # 2*[2], 4*[0.5], 4*[0.5], [0.2], # circles
                      2*[3], 4*[0.2], [0.2], # circles
                      2*[3], 4*[0.2], [0.2], # diag circles
                      4*[0.2], 4*[0.1]]  # stops
        probs = np.array([p for ps in probs_list for p in ps])
        probs = probs / np.sum(probs)
        
        
        tile_sketches = [{}]
        for i in range(1, probs.shape[0]):
            tile_sketches.append({"main": self.new_tile_sketch()}) 
        
        i_forward = 1
        tile_sketches[i_forward]["main"].line(0.0, -0.5, 0.0, 0.5)
        
        i_diag = i_forward + 2
        tile_sketches[i_diag]["main"].line(0.5, -0.5, -0.5, 0.5)
        
        i_eight_turn_up = i_diag + 2
        x = turn_radius / (np.sqrt(2) + 1)
        a = x / np.sqrt(2)
        tile_sketches[i_eight_turn_up]["main"].arc(-x, -turn_radius, 2*turn_radius, 2*turn_radius, 1.5*np.pi, 1.75 * np.pi)
        tile_sketches[i_eight_turn_up]["main"].line(-0.5, 0.0, -x, 0.0)
        tile_sketches[i_eight_turn_up]["main"].line(a, -a, 0.5, -0.5)
        
        i_eight_turn_down = i_eight_turn_up + 4
        tile_sketches[i_eight_turn_down]["main"].arc(-x, turn_radius, 2*turn_radius, 2*turn_radius, 0.25*np.pi, 0.5 * np.pi)
        tile_sketches[i_eight_turn_down]["main"].line(-0.5, 0.0, -x, 0.0)
        tile_sketches[i_eight_turn_down]["main"].line(a, a, 0.5, 0.5)
        
        i_quarter_turn = i_eight_turn_down + 4
        quarter_turn_radius = 0.5 * turn_radius
        tile_sketches[i_quarter_turn]["main"].arc(-quarter_turn_radius, -quarter_turn_radius, 2*quarter_turn_radius, 2*quarter_turn_radius, 1.5*np.pi, 2.0 * np.pi)
        tile_sketches[i_quarter_turn]["main"].line(-0.5, 0.0, -quarter_turn_radius, 0.0)
        tile_sketches[i_quarter_turn]["main"].line(0.0, -quarter_turn_radius, 0.0, -0.5)
        
        i_circle_forward = i_quarter_turn + 4
        tile_sketches[i_circle_forward]["main"].circle(0, 0, radius=circle_radius)
        tile_sketches[i_circle_forward][Direction.UP] = self.new_tile_sketch()
        tile_sketches[i_circle_forward][Direction.UP].line(0.0, -0.5, 0.0, -circle_radius)
        tile_sketches[i_circle_forward][Direction.DOWN] = self.new_tile_sketch()
        tile_sketches[i_circle_forward][Direction.DOWN].line(0.0, circle_radius, 0.0, 0.5)
    
        # i_circle_quarter_turn = i_circle_forward + 2
        # tile_sketches[i_circle_quarter_turn].line(0.0, -0.5, 0.0, -circle_radius)
        # tile_sketches[i_circle_quarter_turn].line(0.5, 0.0, circle_radius, 0.0)
        # tile_sketches[i_circle_quarter_turn].circle(0, 0, radius=circle_radius)
    
        # i_circle_triple = i_circle_quarter_turn + 4
        i_circle_triple = i_circle_forward + 2
        for dir in tile_sketches[i_circle_forward]:
            tile_sketches[i_circle_triple][dir] = self.new_tile_sketch()
            tile_sketches[i_circle_triple][dir].sketch(tile_sketches[i_circle_forward][dir])
        tile_sketches[i_circle_triple][Direction.RIGHT] = self.new_tile_sketch()
        tile_sketches[i_circle_triple][Direction.RIGHT].line(0.5, 0.0, circle_radius, 0.0)
        
        i_circle_all = i_circle_triple + 4
        for dir in tile_sketches[i_circle_triple]:
            tile_sketches[i_circle_all][dir] = self.new_tile_sketch()
            tile_sketches[i_circle_all][dir].sketch(tile_sketches[i_circle_triple][dir])
        tile_sketches[i_circle_all][Direction.LEFT] = self.new_tile_sketch()
        tile_sketches[i_circle_all][Direction.LEFT].line(-0.5, 0.0, -circle_radius, 0.0)
        
        i_circle_diag_forward = i_circle_all + 1
        tile_sketches[i_circle_diag_forward]["main"].circle(0, 0, radius=circle_radius)
        tile_sketches[i_circle_diag_forward][Direction.LOWER_LEFT] = self.new_tile_sketch()
        tile_sketches[i_circle_diag_forward][Direction.LOWER_LEFT].line(-0.5, 0.5, -0.5 * np.sqrt(2) * circle_radius, 0.5 * np.sqrt(2) * circle_radius)
        tile_sketches[i_circle_diag_forward][Direction.UPPER_RIGHT] = self.new_tile_sketch()
        tile_sketches[i_circle_diag_forward][Direction.UPPER_RIGHT].line(0.5, -0.5, 0.5 * np.sqrt(2) * circle_radius, -0.5 * np.sqrt(2) * circle_radius)
        
        i_circle_diag_triple = i_circle_diag_forward + 2
        for dir in tile_sketches[i_circle_diag_forward]:
            tile_sketches[i_circle_diag_triple][dir] = self.new_tile_sketch()
            tile_sketches[i_circle_diag_triple][dir].sketch(tile_sketches[i_circle_diag_forward][dir])
        tile_sketches[i_circle_diag_triple][Direction.LOWER_RIGHT] = self.new_tile_sketch()
        tile_sketches[i_circle_diag_triple][Direction.LOWER_RIGHT].line(0.5, 0.5, 0.5 * np.sqrt(2) * circle_radius, 0.5 * np.sqrt(2) * circle_radius)
        
        i_circle_diag_all = i_circle_diag_triple + 4
        for dir in tile_sketches[i_circle_diag_triple]:
            tile_sketches[i_circle_diag_all][dir] = self.new_tile_sketch()
            tile_sketches[i_circle_diag_all][dir].sketch(tile_sketches[i_circle_diag_triple][dir])
        tile_sketches[i_circle_diag_all][Direction.UPPER_LEFT] = self.new_tile_sketch()
        tile_sketches[i_circle_diag_all][Direction.UPPER_LEFT].line(-0.5, -0.5, -0.5 * np.sqrt(2) * circle_radius, -0.5 * np.sqrt(2) * circle_radius)
        
        i_stop = i_circle_diag_all + 1
        tile_sketches[i_stop]["main"].line(0.0, 0.0, 0.5, 0.0)
        tile_sketches[i_stop]["main"].line(0.0, 0.5*stop_length, 0.0, -0.5*stop_length)
        
        i_stop_diag = i_stop + 4
        stop_length_diag = stop_length / np.sqrt(2)
        tile_sketches[i_stop_diag]["main"].line(0.0, 0.0, 0.5, -0.5)
        tile_sketches[i_stop_diag]["main"].line(-0.5*stop_length_diag, -0.5*stop_length_diag,
                                        0.5*stop_length_diag, 0.5*stop_length_diag)
        
    
        indices_quarter_rot = []
        indices_quarter_rot.extend([i_forward + 1, i_diag + 1])
        indices_quarter_rot.extend(range(i_eight_turn_up + 1, i_eight_turn_up + 4))
        indices_quarter_rot.extend(range(i_eight_turn_down + 1, i_eight_turn_down + 4))
        indices_quarter_rot.extend(range(i_quarter_turn + 1, i_quarter_turn + 4))
        indices_quarter_rot.extend([i_circle_forward + 1])
        # indices_quarter_rot.extend(range(i_circle_quarter_turn + 1, i_circle_quarter_turn + 4))
        indices_quarter_rot.extend(range(i_circle_triple + 1, i_circle_triple + 4))
        indices_quarter_rot.extend([i_circle_diag_forward + 1])
        indices_quarter_rot.extend(range(i_circle_diag_triple + 1, i_circle_diag_triple + 4))
        indices_quarter_rot.extend(range(i_stop + 1, i_stop + 4))
        indices_quarter_rot.extend(range(i_stop_diag + 1, i_stop_diag + 4))
        
        num_tiles = len(tile_sketches)
        # For this case we let 0:7 = right, upper right, up, upper left, ...
        valid_directions = [[] for _ in range(num_tiles)]
        # valid_directions[0] = [dir for dir in Direction]
        valid_directions[i_forward] = [Direction.UP, Direction.DOWN]
        valid_directions[i_diag] = [Direction.UPPER_RIGHT, Direction.LOWER_LEFT]
        valid_directions[i_eight_turn_up] = [Direction.UPPER_RIGHT, Direction.LEFT]
        valid_directions[i_eight_turn_down] = [Direction.LEFT, Direction.LOWER_RIGHT]
        valid_directions[i_quarter_turn] = [Direction.LEFT, Direction.UP]
        valid_directions[i_circle_forward] = [Direction.UP, Direction.DOWN]
        # valid_directions[i_circle_quarter_turn] = [Direction.RIGHT, Direction.UP]
        valid_directions[i_circle_triple] = [Direction.RIGHT, Direction.UP, Direction.DOWN]
        valid_directions[i_circle_all] = [Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN]
        valid_directions[i_circle_diag_forward] = [Direction.UPPER_RIGHT, Direction.LOWER_LEFT]
        valid_directions[i_circle_diag_triple] = [Direction.UPPER_RIGHT, Direction.LOWER_LEFT, Direction.LOWER_RIGHT]
        valid_directions[i_circle_diag_all] = [Direction.UPPER_RIGHT, Direction.LOWER_LEFT, Direction.UPPER_LEFT, Direction.LOWER_RIGHT]
        valid_directions[i_stop] = [Direction.RIGHT]
        valid_directions[i_stop_diag] = [Direction.UPPER_RIGHT]
        
        directions = [dir for dir in Direction]
        for i in indices_quarter_rot:
            for dir in tile_sketches[i-1].keys():
                if dir in directions:
                    dir_i = Direction((dir.value + 2) % 8)
                else:
                    dir_i = dir  # if not direction, just copy
                tile_sketches[i][dir_i] = self.new_tile_sketch()
                tile_sketches[i][dir_i].rotate(-0.5 * np.pi)
                tile_sketches[i][dir_i].sketch(tile_sketches[i-1][dir])
            
            for direction in valid_directions[i-1]:
                valid_directions[i].append(Direction((direction.value + 2) % 8))
        
        
        tiles = []
        for index, (prob, valid_dirs, sketches) in enumerate(zip(probs, valid_directions, tile_sketches)):
            tiles.append(Tile(index, prob, valid_dirs, sketches)) 
        
        # Generate all rules based on valid_directions:
        ruleset = [[] for i in range(len(tiles))]
        for i, valid_dirs in enumerate(valid_directions):  # for valid dirs in tile i
            for j, other_valid_dirs in enumerate(valid_directions):  # for valid dirs in tile j
                for dir in Direction:
                    opposite_dir = reverse_dir(dir)
                    # "If tile i connects to the right and tile j does not connect to the left, the relation is illegal"
                    if (dir in valid_dirs and opposite_dir not in other_valid_dirs) or \
                        (dir not in valid_dirs and opposite_dir in other_valid_dirs):
                            ruleset[i].append(Rule(tiles[i], dir, tiles[j], must_be=False))

            # "if tile has dir upper right, then tile to the right cannot have dir upper left,"
            # and tile above cannot have dir lower right"
            # This means to again check all combinations, and to take the diagonal dirs only, find the two around it 
            # and check if they have valid dirs that points towards the same point 
            for dir in valid_dirs:
                if dir in [Direction.UPPER_RIGHT, Direction.UPPER_LEFT, Direction.LOWER_RIGHT, Direction.LOWER_LEFT]:
                    dir_prev = Direction((dir.value - 3 - 4) % 8)
                    dir_prev_diag = Direction((dir.value - 2) % 8)
                        
                    dir_next = Direction((dir.value + 3 + 4) % 8)
                    dir_next_diag = Direction((dir.value + 2) % 8)

                    for j, other_valid_dirs in enumerate(valid_directions):  # for valid dirs in tile j
                        if dir_prev_diag in other_valid_dirs:
                            # print(Rule(tiles[i], dir_prev, tiles[j], must_be=False))
                            ruleset[i].append(Rule(tiles[i], dir_prev, tiles[j], must_be=False))
                        if dir_next_diag in other_valid_dirs:
                            # print(Rule(tiles[i], dir_next, tiles[j], must_be=False))
                            ruleset[i].append(Rule(tiles[i], dir_next, tiles[j], must_be=False))
                
        return tiles, valid_directions, ruleset
                
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=True)
        vsk.scale("cm")
        vsk.scale(self.tile_size, self.tile_size)
        
        if self.debug_grid:
            self.draw_grid(vsk)

        if self.tileset == "knots":
            tileset, valid_directions, ruleset = self.generate_knot_tileset(width=self.width, radius=self.radius_circles)
        elif self.tileset == "metro":
            tileset, valid_directions, ruleset = self.generate_metro_tileset(circle_radius=self.radius_circles, 
                                                                             turn_radius=self.radius_turns) 
        
        n_tiles = len(tileset)
        
        
        # TODO: not all rules that make sense to add are added. E.g. diagonals that crash or T11 to the right of T32.
        # Are there any cases where e.g. an up next to a diagonal in up left/right direction makes sense?
        
        # Generate valid tiles on edges:
        if self.connected_edge:
            invalid_edge_tiles = {Direction.RIGHT: [], Direction.UP: [], Direction.LEFT: [], Direction.DOWN: []}
            for i, directions in enumerate(valid_directions):
                for dir in invalid_edge_tiles.keys():
                    dir_prev = Direction((dir.value - 1) % 8)
                    dir_next = Direction((dir.value + 1) % 8)
                    # "If right has upper right, right or lower right as valid directions, the tile is not valid on edge"
                    if dir in directions or dir_prev in directions or dir_next in directions:
                        invalid_edge_tiles[dir].append(i)
        else:
            invalid_edge_tiles = None
        
        valid_dirs = valid_directions if self.debug_valid_dirs else None
        if self.debug_tiles_order:
            map = np.arange(self.n_rows * self.n_cols).reshape((self.n_rows, self.n_cols)).astype(float)
            map[map >= n_tiles] = np.nan
            draw_map(map, tileset, vsk, debug_tile_indices=self.debug_tile_indices, debug_cell_indices=self.debug_cell_indices, 
                     valid_dirs=valid_dirs, size=self.tile_size)
        else:
            init_map = None
            if self.use_custom_init:
                init_map = np.full((self.n_rows, self.n_cols), np.nan)
                init_map[4, 0] = 31
                init_map[1, 9] = 33
                init_map[9, 7] = 32
                # TODO: do this proper
                
            
            wfc = WFC(tileset, ruleset, self.n_rows, self.n_cols, invalid_edge_tiles=invalid_edge_tiles, init_map=init_map, 
                      debug=self.debug_print)
            wfc.solve()
            wfc.draw(vsk, debug_tile_indices=self.debug_tile_indices, debug_cell_indices=self.debug_cell_indices, 
                     valid_dirs=valid_dirs, layers=wfc.map_layers, size=self.tile_size)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    WfcSketch.display()
