import vsketch
import numpy as np
from graphlib import TopologicalSorter
import random

# Ideas:
# - isometric grid with towers, bridges, slopes etc.
# - 3d tubes using circles

# We define grid and box like this:
# \             /
#  \    / \    /
#   \ y \ / x /
#    \   o   /
#     \     /
#    y \   / x
#       \ /

# TODO: pi / 6 constant

def compute_horizontal_dist(x_iso, y_iso):
    return (x_iso - y_iso) * np.cos(np.pi / 6)

def compute_vertical_dist(x_iso, y_iso):
    return (x_iso + y_iso) * 0.5

def iso_to_screen(x_iso, y_iso):
    return compute_horizontal_dist(x_iso, y_iso), - compute_vertical_dist(x_iso, y_iso)

def euc_3d_to_iso(x, y, z):
    return (x + z), (y + z)

class isoShape:
    def __init__(self, x, y, z, x_size, y_size, z_size) -> None:
        self.x_euc, self.y_euc, self.z_euc = x, y, z
        self.x_size, self.y_size, self.z_size = x_size, y_size, z_size
        self.set_iso_from_euc(x, y, z)
        self.set_min_max_values()
        
    def set_iso_from_euc(self, x, y, z):
        self.x_iso, self.y_iso = euc_3d_to_iso(x, y, z)
        self.dist_h = compute_horizontal_dist(self.x_iso, self.y_iso)
        self.dist_v = compute_vertical_dist(self.x_iso, self.y_iso)
        
    def set_min_max_values(self):
        self.x_min = self.x_iso
        self.y_min = self.y_iso
        self.x_max = self.x_iso + self.x_size + 0.5 * self.z_size / np.cos(np.pi / 2 - np.pi / 6)
        self.y_max = self.y_iso + self.y_size + 0.5 * self.z_size / np.cos(np.pi / 2 - np.pi / 6)
        self.h_min = compute_horizontal_dist(self.x_iso, self.y_iso + self.y_size)
        self.h_max = compute_horizontal_dist(self.x_iso + self.x_size, self.y_iso)
        self.v_min = compute_vertical_dist(self.x_iso, self.y_iso)
        self.v_max = compute_vertical_dist(*euc_3d_to_iso(self.x_euc + self.x_size, self.y_euc + self.y_size, self.z_euc + self.z_size))
    
    def draw(self, vsk):
        lower_bottom = iso_to_screen(*euc_3d_to_iso(self.x_euc, self.y_euc, self.z_euc))
        right_bottom = iso_to_screen(*euc_3d_to_iso(self.x_euc + self.x_size, self.y_euc, self.z_euc))
        left_bottom = iso_to_screen(*euc_3d_to_iso(self.x_euc, self.y_euc + self.y_size, self.z_euc))
        lower_top = iso_to_screen(*euc_3d_to_iso(self.x_euc, self.y_euc, self.z_euc + self.z_size))
        left_top = iso_to_screen(*euc_3d_to_iso(self.x_euc, self.y_euc + self.y_size, self.z_euc + self.z_size))
        right_top = iso_to_screen(*euc_3d_to_iso(self.x_euc + self.x_size, self.y_euc, self.z_euc + self.z_size))
        upper_top = iso_to_screen(*euc_3d_to_iso(self.x_euc + self.x_size, self.y_euc + self.y_size, self.z_euc + self.z_size))
        vsk.polygon([lower_bottom, right_bottom, right_top, upper_top, left_top, left_bottom], close=True) 
        vsk.line(*lower_bottom, *lower_top)       
        vsk.line(*lower_top, *left_top)       
        vsk.line(*lower_top, *right_top)       
    
    def draw_debug(self, vsk, offset=0):
        x_x_min, y_x_min = iso_to_screen(self.x_min, -offset)
        x_x_max, y_x_max = iso_to_screen(self.x_max, -offset)
        x_y_min, y_y_min = iso_to_screen(-offset, self.y_min)
        x_y_max, y_y_max = iso_to_screen(-offset, self.y_max)
        
        vsk.stroke(2)
        vsk.strokeWeight(3)
        vsk.line(x_x_min, y_x_min, x_x_max, y_x_max)
        vsk.line(x_y_min, y_y_min, x_y_max, y_y_max)
        vsk.line(6 + offset, -self.v_min, 6 + offset, -self.v_max)
        vsk.line(self.h_min, -7 - offset, self.h_max, -7 - offset)
        vsk.stroke(1)
        vsk.strokeWeight(1)
        
def draw_axes(vsk, x_axis_length=5, y_axis_length=5):
    x_x, y_x = iso_to_screen(x_axis_length, 0)
    x_y, y_y = iso_to_screen(0, y_axis_length)
    vsk.line(0, 0, x_x, y_x)
    vsk.line(0, 0, x_y, y_y)

def draw_grid(vsk, grid_size=1, x_size=5, y_size=5):
    for x in np.arange(0, x_size + 1, grid_size):
        vsk.line(*iso_to_screen(x, 0), *iso_to_screen(x, y_size))
    for y in np.arange(0, y_size + 1, grid_size):
        vsk.line(*iso_to_screen(0, y), *iso_to_screen(x_size, y))
        
def check_iso_shape_overlap(shape_1, shape_2):
    return not (shape_1.x_min >= shape_2.x_max or shape_2.x_min >= shape_1.x_max) and \
           not (shape_1.y_min >= shape_2.y_max or shape_2.y_min >= shape_1.y_max) and \
           not (shape_1.h_min >= shape_2.h_max or shape_2.h_min >= shape_1.h_max)

def check_iso_shape_in_front(shape_1, shape_2):
    '''Note: assumes shape_1 and shape_2 overlap.'''
    if shape_1.x_euc >= shape_2.x_euc + shape_2.x_size: return False
    elif shape_2.x_euc >= shape_1.x_euc + shape_1.x_size: return True
    
    if shape_1.y_euc >= shape_2.y_euc + shape_2.y_size: return False
    elif shape_2.y_euc >= shape_1.y_euc + shape_1.y_size: return True
    
    if shape_1.z_euc >= shape_2.z_euc + shape_2.z_size: return True
    elif shape_2.z_euc >= shape_1.z_euc + shape_1.z_size: return False

def get_draw_order(shapes):
    # Need to construct graph and run topological sort
    graph = {}
    for i, shape in enumerate(shapes):
        graph[str(i)] = []  # new empty array for current shape
        
        for node_key in graph.keys():  # for every node already visited
        # create array of shapes to "visit" (all the ones that are already in graph)
            if node_key != str(i):
                node_shape = shapes[int(node_key)]
                if check_iso_shape_overlap(shape, node_shape):  # for every one check if they overlap and if so which is in front. Create corresponding edge in graph.
                    if check_iso_shape_in_front(shape, node_shape):
                        graph[node_key].append(str(i))
                    else:
                        graph[str(i)].append(node_key)
                        
        # print(graph)
        
    ts = TopologicalSorter(graph)
    return [int(i) for i in reversed([*ts.static_order()])]

class IsoSketch(vsketch.SketchClass):
    
    draw_axes = vsketch.Param(False)
    draw_grid = vsketch.Param(True)
    draw_debug = vsketch.Param(False)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        # DO NO OCCULT THINGS HERE
        
        vsk.vpype("occult -i")
        
        if self.draw_axes: draw_axes(vsk, x_axis_length=7, y_axis_length=7)
        if self.draw_grid: draw_grid(vsk, x_size=6, y_size=6)
        
        shape_0 = isoShape(2, 1, 0, 2, 1, 2)
        shape_1 = isoShape(1, 2, 0, 3, 1, 1)
        shape_2 = isoShape(1, 4, 0, 1, 1, 1)
        shape_3 = isoShape(2, 3, 0, 1, 2, 1)
        shapes = [shape_0, shape_1, shape_2, shape_3]
        # random.shuffle(shapes)
        
        # shape_1.draw_debug(vsk, offset=0.2)
        
        # print(check_iso_shape_overlap(shape_1, shape_2))
        # print(check_iso_shape_overlap(shape_2, shape_3))
        # print(check_iso_shape_in_front(shape_1, shape_2))
        # print(check_iso_shape_in_front(shape_2, shape_1))
        
        draw_order = get_draw_order(shapes)
        # print(draw_order)
        
        for i in draw_order:
            shapes[i].draw(vsk)
            if self.draw_debug: shapes[i].draw_debug(vsk, offset=0.2*i)
            
        vsk.vpype("occult -i")
        
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    IsoSketch.display()
