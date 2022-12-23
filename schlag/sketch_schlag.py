import vsketch
import numpy as np
from numpy.random import default_rng
# from shapely.geometry import Polygon
from enum import Enum

def pick_random_element(probs):
    return np.random.choice(len(probs), p=probs)

def enum_type_to_int(enum_type):
    return enum_type.value - 1

def get_empty_sketch(detail=1e-2):
    sketch = vsketch.Vsketch()
    sketch.detail(detail)
    return sketch


def draw_filled_circle(x, y, radius, line_width=1e-2):
    sketch = get_empty_sketch()
    N = int(radius / line_width)
    for r in np.linspace(radius, 0, N):
        sketch.circle(x, y, radius=r)
    return sketch


def draw_circle(x, y, radius):  
    sketch = get_empty_sketch()
    sketch.circle(x, y, radius=radius)
    return sketch

        
def draw_shaded_circle(x_0, y_0, radius, fill_distance, angle=0.0):
    sketch = get_empty_sketch()
    sketch.circle(x_0, y_0, radius=radius)
    N = np.max((0, int(np.round(2 * (radius / fill_distance - 1)))))
    fill_distance = 2 * radius / (N + 1)
    sketch.translate(x_0, y_0)
    sketch.rotate(angle)
    for d in np.linspace(-radius + fill_distance, radius - fill_distance, N, endpoint=True):
        dy = radius * np.sin(np.arccos(d / radius))
        sketch.line(d, -dy, d, dy) 
    return sketch
            

def draw_shaded_rect(x, y, width, height, fill_distance, angle=0.0):
    sketch = get_empty_sketch()
    sketch.rect(x, y, width, height, mode="center")
    N = np.max((0, int(np.round(height / fill_distance - 1))))
    fill_distance = height / (N + 1)
    sketch.translate(x, y)
    for y in np.linspace(-0.5*height + fill_distance, 0.5*height - fill_distance, N, endpoint=True):
        sketch.line(-0.5*width, y, 0.5*width, y)
    # TODO: add arbitrary shading angle 
    return sketch


def draw_filled_rect(x, y, width, height, angle=0.0):
    sketch = draw_shaded_rect(x, y, width, height, fill_distance=1e-2, angle=angle)
    return sketch


def draw_rect(x, y, width, height):
    sketch = get_empty_sketch()
    sketch.rect(x, y, width, height, mode="center")
    return sketch
    

def draw_dotted_circle(x, y, radius, radius_inner):
    sketch = get_empty_sketch()
    sketch.circle(x, y, radius=radius)
    sketch.sketch(draw_filled_circle(x, y, radius=radius_inner))
    return sketch
    

class SchlagSketch(vsketch.SketchClass):
    # General params:
    N_grid_x = vsketch.Param(50, min_value=1)
    N_grid_y = vsketch.Param(50, min_value=1)
    width = vsketch.Param(10.0)
    scale = vsketch.Param(1.0, min_value=0.0)
    debug_show_blob = vsketch.Param(True)
    debug_show_blob_shapes = vsketch.Param(True)
    debug_show_shapes = vsketch.Param(True)
    occult = vsketch.Param(False)
    padding = vsketch.Param(0.20)
    
    
    # Occupancy grid generation params:
    # Metaballs:
    N_metaballs = vsketch.Param(10, min_value=0)
    r_metaballs_min = vsketch.Param(0.3, min_value=0.0)
    r_metaballs_max = vsketch.Param(1.2, min_value=0.0)
    metaballs_thresh = vsketch.Param(1.0, min_value=0.0)
    
    # Rects:
    N_rects_blob_min = vsketch.Param(1, min_value=0)
    N_rects_blob_max = vsketch.Param(5, min_value=0)
    rects_blob_width_min = vsketch.Param(0.4, min_value=0)
    rects_blob_width_max = vsketch.Param(2.0, min_value=0)
    rects_blob_height_gain_max = vsketch.Param(4.0, min_value=0)
    
    # Circles:
    N_circles_blob_min = vsketch.Param(1, min_value=0)
    N_circles_blob_max = vsketch.Param(5, min_value=0)
    circles_blob_radius_min = vsketch.Param(0.3, min_value=0)
    circles_blob_radius_max = vsketch.Param(1.0, min_value=0)
    
    
    # Shape params:
    N_shapes = vsketch.Param(100, min_value=0)
    shapes = Enum("Shape", "SHADED_RECT SHADED_CIRCLE")
    # p_rect = vsketch.Param(0.5, min_value=0, max_value=1)
    # p_circle = vsketch.Param(0.5, min_value=0, max_value=1)
    # p_filled_rect = vsketch.Param(0.5, min_value=0, max_value=1)
    # p_filled_circle = vsketch.Param(0.5, min_value=0, max_value=1)
    p_shaded_rect = vsketch.Param(0.5, min_value=0, max_value=1)
    p_shaded_circle = vsketch.Param(0.5, min_value=0, max_value=1)
    
    rect_width_min = vsketch.Param(0.1, min_value=0)
    rect_width_max = vsketch.Param(0.8, min_value=0)
    rect_height_gain_max = vsketch.Param(4.0, min_value=0)
    
    circle_radius_min = vsketch.Param(0.2, min_value=0)
    circle_radius_max = vsketch.Param(0.6, min_value=0)
    
    fill_distance_min = vsketch.Param(0.04, min_value=0)
    fill_distance_max = vsketch.Param(0.25, min_value=0)
    
    
    def check_valid_pos(self, x, y, grid):
        return (0 <= x < grid.shape[0]) and (0 <= y < grid.shape[1])
    
    def sample_random_occupied_pos(self, x_occupied, y_occupied, unit_dist=None):
        N_valid = x_occupied.shape[0]        
        choice_index = self.rng.integers(0, N_valid)    
        x, y = self.xs_grid[x_occupied[choice_index]], self.ys_grid[y_occupied[choice_index]]
        if unit_dist is not None:
            dx, dy = self.rng.uniform(-0.5*unit_dist, 0.5*unit_dist, 2)
            x, y = x + dx, y + dy
        return x, y
    
    def generate_metaballs_blob(self, xy_metaballs, r_metaballs):
        f_grid = np.zeros((self.N_grid_x, self.N_grid_y))
        occupancy_grid = np.full((self.N_grid_x, self.N_grid_y), False)
        for i, x in enumerate(self.xs_grid):
            for j, y in enumerate(self.ys_grid):
                d = np.array([x,y]) - xy_metaballs
                f = np.sum(r_metaballs**2 / (d[:,0]**2 + d[:,1]**2 + 1e-6))
                f_grid[i,j] = f
                
                if f_grid[i,j] > self.metaballs_thresh:
                    occupancy_grid[i,j] = True
        return occupancy_grid
    
    def add_rect_to_grid(self, grid, x_0, y_0, width, height, angle, unit_dist):
        N_x = int(np.ceil(1.1 * np.sqrt(2) * width / unit_dist))
        N_y = int(np.ceil(1.1 * np.sqrt(2) * height / unit_dist))
        for x in np.linspace(-0.5*width, 0.5*width, N_x):
            for y in np.linspace(-0.5*height, 0.5*height, N_y):
                x_paper = x_0 + x * np.cos(angle) + y * np.sin(angle)
                y_paper = y_0 + x * np.sin(angle) - y * np.cos(angle)
                x_index = int(np.round(x_paper / unit_dist))
                y_index = int(np.round(y_paper / unit_dist))
                if self.check_valid_pos(x_index, y_index, grid):
                    grid[x_index,y_index] = True
        return grid
    
    def add_circle_to_grid(self, grid, x_0, y_0, radius, unit_dist):
        N_x = int(np.ceil(4 * radius / unit_dist))
        for x in np.linspace(-radius, radius, N_x):
            y_lim = radius * np.sin(np.arccos(x / radius))
            N_y = int(np.ceil(4 * y_lim / unit_dist))
            for y in np.linspace(-y_lim, y_lim, N_y):
                x_index = int(np.round((x + x_0) / unit_dist))
                y_index = int(np.round((y + y_0) / unit_dist))
                if self.check_valid_pos(x_index, y_index, grid):
                    grid[x_index,y_index] = True
        return grid

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale)
        
        self.p_shapes = np.array([self.p_shaded_rect, self.p_shaded_circle])
        
        self.unit_size = self.width / self.N_grid_x
        self.height = self.N_grid_y * self.unit_size
        self.xs_grid = np.linspace(0, self.width, num=self.N_grid_x)
        self.ys_grid = np.linspace(0, self.height, num=self.N_grid_y)
        
        self.rng = default_rng()
        xy_metaballs_min = np.full(2, self.r_metaballs_max) + self.padding
        xy_metaballs_max = np.array([self.width, self.height]) - xy_metaballs_min
        xy_metaballs = self.rng.uniform(low=xy_metaballs_min, high=xy_metaballs_max, size=(self.N_metaballs,2))
        r_metaballs = self.rng.uniform(low=self.r_metaballs_min, high=self.r_metaballs_max, size=self.N_metaballs)
        
        # Compute occupancy grid to generate metaballs blob:
        occupancy_grid = self.generate_metaballs_blob(xy_metaballs, r_metaballs)
        x_valid, y_valid = np.where(occupancy_grid)
        
        # Add rects to blob:
        N_rects = self.rng.integers(self.N_rects_blob_min, self.N_rects_blob_max + 1)
        for i in range(N_rects):
            x_rect, y_rect = self.sample_random_occupied_pos(x_valid, y_valid)
            rect_angle = self.rng.uniform(0, np.pi)
            rect_width = self.rng.uniform(self.rects_blob_width_min, self.rects_blob_width_max)
            rect_height = rect_width * self.rng.uniform(1.0, self.rects_blob_height_gain_max)  # random orientation so can do this to simplify tuning
            self.add_rect_to_grid(occupancy_grid, x_rect, y_rect, rect_width, rect_height, rect_angle, unit_dist=self.unit_size)
            if self.debug_show_blob_shapes:
                with vsk.pushMatrix():
                    vsk.translate(x_rect, y_rect)
                    vsk.rotate(rect_angle)
                    vsk.rect(0, 0, rect_width, rect_height, mode="center")
                    
        # Add circles to blob:
        N_circles = self.rng.integers(self.N_circles_blob_min, self.N_circles_blob_max + 1)
        for i in range(N_circles):
            x_circle, y_circle = self.sample_random_occupied_pos(x_valid, y_valid)
            radius = self.rng.uniform(self.circles_blob_radius_min, self.circles_blob_radius_max)
            self.add_circle_to_grid(occupancy_grid, x_circle, y_circle, radius, unit_dist=self.unit_size)
            if self.debug_show_blob_shapes:
                vsk.circle(x_circle, y_circle, radius=radius)

        x_valid, y_valid = np.where(occupancy_grid)  # update array of occupied cells after adding other shapes 
        
        
        # Draw shapes:
        # TODO: scale size over time
        # TODO: add boundary only shapes
        vsk.stroke(4)
        for i in range(self.N_shapes):
            choice = pick_random_element(self.p_shapes)
            x, y = self.sample_random_occupied_pos(x_valid, y_valid, unit_dist=self.unit_size)
            angle = self.rng.uniform(0, np.pi)
            if choice == enum_type_to_int(self.shapes.SHADED_RECT):
                width = self.rng.uniform(self.rect_width_min, self.rect_width_max)
                height = width * self.rng.uniform(1.0, self.rect_height_gain_max)
                fill_distance = self.rng.uniform(self.fill_distance_min, self.fill_distance_max)
                rect = draw_shaded_rect(0, 0, width, height, fill_distance)
                with vsk.pushMatrix():
                    vsk.translate(x, y)
                    vsk.rotate(angle)
                    vsk.sketch(rect)
            if choice == enum_type_to_int(self.shapes.SHADED_CIRCLE):
                radius = self.rng.uniform(self.circle_radius_min, self.circle_radius_max)
                fill_distance = self.rng.uniform(self.fill_distance_min, self.fill_distance_max)
                vsk.sketch(draw_shaded_circle(x, y, radius, fill_distance, angle))

        
        # Debug draws:
        if self.debug_show_blob_shapes:
            vsk.stroke(1)
            for xy, r in zip(xy_metaballs, r_metaballs):
                vsk.circle(xy[0], xy[1], radius=r)
            
        if self.debug_show_blob:
            vsk.stroke(2)
            for i, x in enumerate(self.xs_grid):
                for j, y in enumerate(self.ys_grid):
                    vsk.circle(x, y, radius=0.01)
                    if occupancy_grid[i,j]:
                        vsk.circle(x, y, radius=0.5*self.unit_size)
                
        if self.debug_show_shapes:
            vsk.stroke(3)
            vsk.sketch(draw_shaded_circle(0, 0, radius=0.375, fill_distance=0.1, angle=np.deg2rad(45)))
            vsk.sketch(draw_shaded_rect(1, 0, width=0.7, height=0.5, fill_distance=0.1))
            vsk.sketch(draw_dotted_circle(2, 0, radius=0.375, radius_inner=0.15))
            vsk.sketch(draw_filled_circle(3, 0, radius=0.375))
            vsk.sketch(draw_circle(4, 0, radius=0.375))
            vsk.sketch(draw_rect(5, 0, 0.7, 0.5))
            vsk.sketch(draw_filled_rect(6, 0, 0.7, 0.5))
            # vsk.triangle()
            

        if self.occult:
            vsk.vpype("occult -i")
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    SchlagSketch.display()
