import vsketch
import numpy as np
from graphlib import TopologicalSorter


# We define grid and box like this:
# \             /
#  \    / \    /
#   \ y \ / x /
#    \   o   /
#     \     /
#    y \   / x
#       \ /

# TODO: pi / 6 constant

# TODO: connected textures (see explanation iso_town)
# how to do this with occult? only way I see is to add polygon to other layer and only draw the first layer?
# only way I know... 
# No, that also doesnt work because everythign on layer 2 is above layer 1...

# TODO: Slopes and corner slopes

# TODO: somehow to basic view culling or whatever it is called, basically find out which shapes are in view and only draw those.
#   for all shapes:
#       have set of sets that describe what is being covered of x, y, v
#           for all shapes in front of shape:
#               add x, y, v coverage set of shape to the set of sets.
#               if completely covered: break and mark as "do not draw".
#       mark as "draw"

# main question here is how to define set. Pythons own set doesnt really work.
# python-ranges library does what we want. But could also be implemented from scratch. 
# With that lib we can simply do Union of the preexsisting RangeSet for x, y, v with a new Range for the visited shape.
# And to check if completely covered can either check if intersection is equal to the shape bounds 
# or possibly the in operator works directly? Need to check this.

# TODO: add different shapes:
# - slopes
# - corner slopes
# - roof triangles
# - pyramids
# - arches
# - stairs 
# - 2d wall textures, e.g. windows and doors
# - sphere (ellipse in iso)
# - dome

# larger building blocks:
# - chimneys
# - handrails
# - bridges

def compute_horizontal_dist(x_iso, y_iso):
    return (x_iso - y_iso) * np.cos(np.pi / 6)


def compute_vertical_dist(x_iso, y_iso):
    return (x_iso + y_iso) * 0.5


def iso_to_screen(x_iso, y_iso, scale=1):
    return scale*compute_horizontal_dist(x_iso, y_iso), -scale*compute_vertical_dist(x_iso, y_iso)


def euc_3d_to_iso(x, y, z):
    return (x + z), (y + z)


class isoShape:
    def __init__(self, x, y, z, x_size, y_size, z_size, scale=1) -> None:
        self.x_euc, self.y_euc, self.z_euc = x, y, z
        self.x_size, self.y_size, self.z_size = x_size, y_size, z_size
        self.scale = scale
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
    
    
    def draw(self, vsk, dx_shade=None, dy_shade=None, dz_shade=None):
        raise Exception(NotImplementedError)
    
    def draw_debug(self, vsk, offset=0):
        x_x_min, y_x_min = iso_to_screen(self.x_min, -offset, self.scale)
        x_x_max, y_x_max = iso_to_screen(self.x_max, -offset, self.scale)
        x_y_min, y_y_min = iso_to_screen(-offset, self.y_min, self.scale)
        x_y_max, y_y_max = iso_to_screen(-offset, self.y_max, self.scale)
        
        vsk.stroke(2)
        vsk.strokeWeight(3)
        vsk.line(x_x_min, y_x_min, x_x_max, y_x_max)
        vsk.line(x_y_min, y_y_min, x_y_max, y_y_max)
        vsk.line(6 + offset, -self.v_min, 6 + offset, -self.v_max)
        vsk.line(self.h_min, -7 - offset, self.h_max, -7 - offset)
        vsk.stroke(1)
        vsk.strokeWeight(1)


class Box(isoShape):
    def __init__(self, x, y, z, x_size, y_size, z_size, scale=1) -> None:
        super().__init__(x, y, z, x_size, y_size, z_size, scale)       
    
    def draw(self, vsk, dx_shade=None, dy_shade=None, dz_shade=None):
        # Get all points in screen coordinates:
        lower_bottom = iso_to_screen(*euc_3d_to_iso(self.x_euc, self.y_euc, self.z_euc), self.scale)
        right_bottom = iso_to_screen(*euc_3d_to_iso(self.x_euc + self.x_size, self.y_euc, self.z_euc), self.scale)
        left_bottom = iso_to_screen(*euc_3d_to_iso(self.x_euc, self.y_euc + self.y_size, self.z_euc), self.scale)
        lower_top = iso_to_screen(*euc_3d_to_iso(self.x_euc, self.y_euc, self.z_euc + self.z_size), self.scale)
        left_top = iso_to_screen(*euc_3d_to_iso(self.x_euc, self.y_euc + self.y_size, self.z_euc + self.z_size), self.scale)
        right_top = iso_to_screen(*euc_3d_to_iso(self.x_euc + self.x_size, self.y_euc, self.z_euc + self.z_size), self.scale)
        upper_top = iso_to_screen(*euc_3d_to_iso(self.x_euc + self.x_size, self.y_euc + self.y_size, self.z_euc + self.z_size), self.scale)
        
        # Draw main polygon of outer shape:
        vsk.stroke(1)
        vsk.polygon([lower_bottom, right_bottom, right_top, upper_top, left_top, left_bottom], close=True)
        
        # Draw inner lines:
        # vsk.stroke(2)
        vsk.line(*lower_bottom, *lower_top)       
        vsk.line(*lower_top, *left_top)       
        vsk.line(*lower_top, *right_top)   
        
        # Draw outer lines:
        # vsk.line(*lower_bottom, *left_bottom)
        # vsk.line(*lower_bottom, *right_bottom)   
        # vsk.line(*left_bottom, *left_top)   
        # vsk.line(*right_bottom, *right_top)   
        # vsk.line(*left_top, *upper_top)   
        # vsk.line(*right_top, *upper_top)   
        # vsk.stroke(1)
        
        
        # Shading:
        if dy_shade is not None:
            y_shade = int(self.y_size / dy_shade)
            dy = self.y_size / (y_shade + 1)
            for y in np.linspace(self.y_euc + dy, self.y_euc + self.y_size - dy, y_shade):
                vsk.line(*iso_to_screen(*euc_3d_to_iso(self.x_euc, y, self.z_euc), self.scale), *iso_to_screen(*euc_3d_to_iso(self.x_euc, y, self.z_euc + self.z_size), self.scale))
    
        if dx_shade is not None:
            x_shade = int(self.x_size / dx_shade)
            dx = self.x_size / (x_shade + 1)
            for x in np.linspace(self.x_euc + dx, self.x_euc + self.x_size - dx, x_shade):
                vsk.line(*iso_to_screen(*euc_3d_to_iso(x, self.y_euc, self.z_euc), self.scale), *iso_to_screen(*euc_3d_to_iso(x, self.y_euc, self.z_euc + self.z_size), self.scale))
        
        if dz_shade is not None:
            z_shade = int(self.y_size / dz_shade)
            dz = self.y_size / (z_shade + 1)
            for y in np.linspace(self.y_euc + dz, self.y_euc + self.y_size - dz, z_shade):
                vsk.line(*iso_to_screen(*euc_3d_to_iso(self.x_euc, y, self.z_euc + self.z_size), self.scale), *iso_to_screen(*euc_3d_to_iso(self.x_euc + self.x_size, y, self.z_euc + self.z_size), self.scale))
     
        
def draw_axes(vsk, x_axis_length=5, y_axis_length=5, scale=1):
    x_x, y_x = iso_to_screen(x_axis_length, 0, scale)
    x_y, y_y = iso_to_screen(0, y_axis_length, scale)
    vsk.line(0, 0, x_x, y_x)
    vsk.line(0, 0, x_y, y_y)


def draw_grid(vsk, grid_size=1, x_size=5, y_size=5, scale=1):
    for x in np.arange(0, x_size + 1, grid_size):
        vsk.line(*iso_to_screen(x, 0, scale), *iso_to_screen(x, y_size, scale))
    for y in np.arange(0, y_size + 1, grid_size):
        vsk.line(*iso_to_screen(0, y, scale), *iso_to_screen(x_size, y, scale))
        
        
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
                        
    ts = TopologicalSorter(graph)
    return [int(i) for i in reversed([*ts.static_order()])]
