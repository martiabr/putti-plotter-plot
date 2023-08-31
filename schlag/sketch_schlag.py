import vsketch
import numpy as np
from numpy.random import default_rng
from enum import Enum

# TODO:
# add boundary only shapes (x)
# lines (x)
# more custom params
# short random angle line shading 
# draw border things before all other things?


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

        
def draw_shaded_circle(x_0, y_0, radius, fill_distance, angle=0.0, fill_gain=1.0):
    sketch = get_empty_sketch()
    sketch.circle(x_0, y_0, radius=radius)
    # N = np.max((0, int(np.round(2 * (radius / fill_distance - 1)))))
    N = np.max((0, int(np.round(fill_gain * 2 * (radius / fill_distance - 1)))))
    fill_distance = fill_gain * 2 * radius / (N + 1)
    start = -radius + fill_distance
    end = -start
    fill_end = start + fill_gain * (end - start)
    sketch.translate(x_0, y_0)
    sketch.rotate(angle)
    for x in np.linspace(start, fill_end, N, endpoint=True):
        dy = radius * np.sin(np.arccos(x / radius))
        sketch.line(x, -dy, x, dy) 
    return sketch


def draw_dot_shaded_circle(x_0, y_0, radius, dot_distance, vsk=None, noise_gain=None, noise_freq=1.0):
    if noise_gain is not None: vsk.noiseSeed(np.random.randint(1e6))
    sketch = get_empty_sketch()
    sketch.translate(x_0, y_0)
    sketch.circle(0, 0, radius=radius)
    sketch.translate(-radius, -radius)
    N = np.max((0, int(np.round(2 * (radius / dot_distance - 1)))))
    dot_distance = 2 * radius / (N + 1)
    for x in np.linspace(0, 2 * radius, N + 2, endpoint=True):
        for y in np.linspace(0, 2 * radius, N + 2, endpoint=True):
            x_noise, y_noise = 0.0, 0.0
            if noise_gain is not None and vsk is not None:
                angle = 2 * np.pi * vsk.noise(noise_freq * x, noise_freq * y)
                x_noise = noise_gain * dot_distance * np.cos(angle)
                y_noise = noise_gain * dot_distance * np.sin(angle)
            x_total, y_total = x + x_noise, y + y_noise
            if (x_total - radius)**2 +(y_total - radius)**2 < radius**2:
                sketch.circle(x_total, y_total, radius=1e-2)
    return sketch


def draw_partial_filled_circle(x, y, radius, fill_gain=0.5, fill_distance=1e-2):
    return draw_shaded_circle(x, y, radius, fill_distance=fill_distance, fill_gain=fill_gain)
     

def draw_shaded_rect(x, y, width, height, fill_distance, angle=0.0):
    sketch = get_empty_sketch()
    sketch.rect(x, y, width, height, mode="center")
    N = np.max((0, int(np.round(height / fill_distance - 1))))
    fill_distance = height / (N + 1)
    sketch.translate(x, y)
    for y in np.linspace(-0.5*height + fill_distance, 0.5*height - fill_distance, N, endpoint=True):
        sketch.line(-0.5*width, y, 0.5*width, y)
    
    # TODO: add arbitrary shading angle
    # fill_distance_y = fill_distance * np.sin(angle)
    # N_y = np.max((0, int(np.round(height / fill_distance_y - 1))))
    # for y in np.linspace(-0.5*height + fill_distance_y, 0.5*height - fill_distance_y, N_y, endpoint=True):
    #     x = (y + 0.5*height) / np.tan(angle) - 0.5*width
    #     sketch.line(x, -0.5*height, -0.5*width, y)
    return sketch


def draw_dot_shaded_rect(x_0, y_0, width, height, dot_distance, vsk=None, noise_gain=None, noise_freq=1.0):
    if noise_gain is not None: vsk.noiseSeed(np.random.randint(1e6))
    sketch = get_empty_sketch()
    sketch.translate(x_0, y_0)
    sketch.rect(0, 0, width, height, mode="center")
    sketch.translate(-0.5 * width, -0.5 * height)
    N_x = np.max((0, int(np.round(width / dot_distance - 1))))
    N_y = np.max((0, int(np.round(height / dot_distance - 1))))
    dot_distance_x = width / (N_x + 1)
    dot_distance_y = height / (N_y + 1)
    for x in np.linspace(0, width, N_x + 2, endpoint=True):
        for y in np.linspace(0, height, N_y + 2, endpoint=True):
            x_noise, y_noise = 0.0, 0.0
            if noise_gain is not None and vsk is not None:
                angle = 2 * np.pi * vsk.noise(noise_freq * x, noise_freq * y)
                x_noise = noise_gain * dot_distance_x * np.cos(angle)
                y_noise = noise_gain * dot_distance_y * np.sin(angle)
            x_total, y_total = x + x_noise, y + y_noise
            if x_total > 0.0 and x_total < width and y_total > 0.0 and y_total < height:
                sketch.circle(x_total, y_total, radius=1e-2)
    return sketch
    

def draw_filled_rect(x, y, width, height, angle=0.0):
    sketch = draw_shaded_rect(x, y, width, height, fill_distance=1e-2, angle=angle)
    return sketch


def draw_rect(x, y, width, height):
    sketch = get_empty_sketch()
    sketch.rect(x, y, width, height, mode="center")
    return sketch
    

def draw_dot_circle(x, y, radius, radius_inner):
    sketch = get_empty_sketch()
    sketch.circle(x, y, radius=radius)
    sketch.sketch(draw_filled_circle(x, y, radius=radius_inner))
    return sketch


def draw_pole(x, y, pole_width, pole_height, radius):
    sketch = get_empty_sketch()
    sketch.translate(x, y -0.5 * pole_height)
    sketch.rect(0, 0, pole_width, pole_height, mode="center")
    sketch.translate(0, -0.5 * (pole_height + radius))
    sketch.circle(0, 0, radius=radius)
    sketch.translate(0, pole_height + 0.5 * radius)
    return sketch


def draw_flag(x, y, pole_width, pole_height, flag_width, flag_height, right=True, triangular=False):
    sketch = get_empty_sketch()
    sketch.translate(x, y - 0.5 * pole_height)
    sketch.rect(0, 0, pole_width, pole_height, mode="center")
    sketch.translate(0, 0.5 * (flag_height - pole_height))

    if triangular:
        sketch.translate(0, -0.5 * flag_height)
        if right:
            sketch.translate(0.5 * pole_width, 0)
            sketch.triangle(0, 0, flag_width, 0.5 * flag_height, 0, flag_height)
        else:
            sketch.translate(-0.5 * pole_width, 0)
            sketch.triangle(0, 0, -flag_width, 0.5 * flag_height, 0, flag_height)
    else:
        if right:
            sketch.translate(0.5 * (flag_width + pole_width), 0)
        else:
            sketch.translate(-0.5 * (flag_width + pole_width), 0)
        sketch.rect(0, 0, flag_width, flag_height, mode="center")
        
    return sketch


def draw_line(x, y, length, width=1e-2):
    sketch = get_empty_sketch()
    sketch.line(x, y, x, y - length)
    return sketch
    

class SchlagSketch(vsketch.SketchClass):
    # General params:
    N_grid_x = vsketch.Param(50, min_value=1)
    N_grid_y = vsketch.Param(50, min_value=1)
    width = vsketch.Param(10.0)
    scale = vsketch.Param(1.0, min_value=0.0)
    debug_show_blob = vsketch.Param(True)
    debug_show_blob_border = vsketch.Param(True)
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
    shapes = Enum("Shape", "SHADED_RECT SHADED_CIRCLE FILLED_RECT FILLED_CIRCLE RECT CIRCLE DOT_CIRCLE DOT_SHADED_RECT " +\
                           "DOT_SHADED_CIRCLE PARTIAL_FILLED_CIRCLE POLE FLAG LINE")
    border_shapes = (shapes.POLE, shapes.FLAG, shapes.LINE)
    p_shaded_rect = vsketch.Param(0.15, min_value=0, max_value=1)
    p_shaded_circle = vsketch.Param(0.15, min_value=0, max_value=1)
    p_filled_rect = vsketch.Param(0.09, min_value=0, max_value=1)
    p_filled_circle = vsketch.Param(0.1, min_value=0, max_value=1)
    p_rect = vsketch.Param(0.05, min_value=0, max_value=1)
    p_circle = vsketch.Param(0.05, min_value=0, max_value=1)
    p_dot_circle = vsketch.Param(0.1, min_value=0, max_value=1)
    p_dot_shaded_rect = vsketch.Param(0.1, min_value=0, max_value=1)
    p_dot_shaded_circle = vsketch.Param(0.1, min_value=0, max_value=1)
    p_partial_filled_circle = vsketch.Param(0.05, min_value=0, max_value=1)
    p_pole = vsketch.Param(0.02, min_value=0, max_value=1)
    p_flag = vsketch.Param(0.02, min_value=0, max_value=1)
    p_line = vsketch.Param(0.02, min_value=0, max_value=1)
    
    do_size_scaling = vsketch.Param(True)
    size_scale_start = vsketch.Param(0.5, min_value=0)
    size_scale_end = vsketch.Param(2.0, min_value=0)
    
    
    rect_width_min = vsketch.Param(0.1, min_value=0)
    rect_width_max = vsketch.Param(0.6, min_value=0)
    rect_height_gain_max = vsketch.Param(3.5, min_value=0)
    
    circle_radius_min = vsketch.Param(0.2, min_value=0)
    circle_radius_max = vsketch.Param(0.6, min_value=0)
    
    fill_distance_min = vsketch.Param(0.04, min_value=0)
    fill_distance_max = vsketch.Param(0.25, min_value=0)
    
    dot_circle_radius_min = vsketch.Param(0.05, min_value=0)
    dot_circle_radius_max = vsketch.Param(0.4, min_value=0)
    dot_circle_inner_radius_gain_min = vsketch.Param(0.1, min_value=0)
    dot_circle_inner_radius_gain_max = vsketch.Param(0.6, min_value=0)
    
    dot_fill_distance_min = vsketch.Param(0.05, min_value=0)
    dot_fill_distance_max = vsketch.Param(0.09, min_value=0)
    p_noisy_dot_fill = vsketch.Param(0.75, min_value=0, max_value=1)
    dot_fill_noise_gain_min = vsketch.Param(0.2, min_value=0)
    dot_fill_noise_gain_max = vsketch.Param(0.6, min_value=0)
    dot_fill_noise_freq_min = vsketch.Param(1.0, min_value=0)
    dot_fill_noise_freq_max = vsketch.Param(6.0, min_value=0)
    
    pole_height_min = vsketch.Param(0.5, min_value=0)
    pole_height_max = vsketch.Param(1.3, min_value=0)
    pole_width_gain_min = vsketch.Param(0.05, min_value=0)
    pole_width_gain_max = vsketch.Param(0.12, min_value=0)
    pole_radius_gain_min = vsketch.Param(1.1, min_value=0)
    pole_radius_gain_max = vsketch.Param(1.5, min_value=0)
    
    p_flag_triangular = vsketch.Param(0.5, min_value=0, max_value=1)
    flag_height_gain_min = vsketch.Param(0.2, min_value=0)
    flag_height_gain_max = vsketch.Param(0.3, min_value=0)
    flag_width_gain_min = vsketch.Param(1.1, min_value=0)
    flag_width_gain_max = vsketch.Param(1.7, min_value=0)
    
    line_length_min = vsketch.Param(0.5, min_value=0)
    line_length_max = vsketch.Param(1.5, min_value=0)

    
    def check_valid_pos(self, x, y, grid):
        return (0 <= x < grid.shape[0]) and (0 <= y < grid.shape[1])
    
    def sample_random_occupied_pos(self, x_occupied, y_occupied, unit_dist=None):
        N_valid = x_occupied.shape[0]        
        choice_index = self.rng.integers(0, N_valid)    
        x, y = self.xs_grid[x_occupied[choice_index]], self.ys_grid[y_occupied[choice_index]]
        if unit_dist is not None:
            dx, dy = self.rng.uniform(-0.5*unit_dist, 0.5*unit_dist, 2)
            x, y = x + dx, y + dy
        return x, y, choice_index
    
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

    def draw_sketch_with_angle(self, vsk, sketch, x, y, angle):
        with vsk.pushMatrix():
            vsk.translate(x, y)
            vsk.rotate(angle)
            vsk.sketch(sketch)

    def draw_debug_shapes(self, vsk):
        vsk.stroke(5)
        with vsk.pushMatrix():
            vsk.translate(0,-2)
            
            for x in range(10):
                for y in range(2):
                    vsk.circle(x, y, radius=3e-2)
                       
            vsk.sketch(draw_shaded_circle(0, 0, radius=0.375, fill_distance=0.1, angle=np.deg2rad(45)))
            vsk.sketch(draw_shaded_rect(1, 0, width=0.7, height=0.5, fill_distance=0.1, angle=np.deg2rad(45)))
            vsk.sketch(draw_dot_circle(2, 0, radius=0.375, radius_inner=0.15))
            vsk.sketch(draw_filled_circle(3, 0, radius=0.375))
            vsk.sketch(draw_filled_rect(4, 0, 0.7, 0.5))
            vsk.sketch(draw_circle(5, 0, radius=0.375))
            vsk.sketch(draw_rect(6, 0, 0.7, 0.5))
            vsk.sketch(draw_pole(7, 0, 0.1, 1, 0.1))
            vsk.sketch(draw_flag(8, 0, 0.1, 1, 0.5, 0.3, right=False, triangular=True))
            vsk.sketch(draw_line(9, 0, 1))
            vsk.sketch(draw_partial_filled_circle(0, 1, 0.375, fill_gain=0.5))
            vsk.sketch(draw_dot_shaded_rect(1, 1, 0.7, 0.6, dot_distance=6e-2))
            vsk.sketch(draw_dot_shaded_rect(2, 1, 0.7, 0.6, dot_distance=6e-2, vsk=vsk, noise_gain=0.4, noise_freq=4.0))
            vsk.sketch(draw_dot_shaded_circle(3, 1, 0.375, dot_distance=6e-2))
            vsk.sketch(draw_dot_shaded_circle(4, 1, 0.375, dot_distance=6e-2, vsk=vsk, noise_gain=0.4, noise_freq=4.0))

    def draw_init(self, vsk):
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale)
        
        self.p_shapes = np.array([self.p_shaded_rect, self.p_shaded_circle, self.p_rect, self.p_circle,
                                  self.p_filled_rect, self.p_filled_circle, self.p_dot_circle,
                                  self.p_dot_shaded_rect, self.p_dot_shaded_circle, self.p_partial_filled_circle,
                                  self.p_pole, self.p_flag, self.p_line])
        
        self.unit_size = self.width / self.N_grid_x
        self.height = self.N_grid_y * self.unit_size
        self.xs_grid = np.linspace(0, self.width, num=self.N_grid_x)
        self.ys_grid = np.linspace(0, self.height, num=self.N_grid_y)
        
        self.rng = default_rng()
    
    def generate_occupancy_grid(self, vsk):
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
            x_rect, y_rect, _ = self.sample_random_occupied_pos(x_valid, y_valid)
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
            x_circle, y_circle, _ = self.sample_random_occupied_pos(x_valid, y_valid)
            radius = self.rng.uniform(self.circles_blob_radius_min, self.circles_blob_radius_max)
            self.add_circle_to_grid(occupancy_grid, x_circle, y_circle, radius, unit_dist=self.unit_size)
            if self.debug_show_blob_shapes:
                vsk.circle(x_circle, y_circle, radius=radius)

        x_valid, y_valid = np.where(occupancy_grid)  # update array of occupied cells after adding other shapes
        
        # Find border:
        border_grid = np.zeros((self.N_grid_x, self.N_grid_y))
        dir_grid = np.zeros((self.N_grid_x, self.N_grid_y, 4))
        for x, y in zip(x_valid, y_valid):
            if y > 0 and not occupancy_grid[x,y-1]:
                border_grid[x,y] = True
                dir_grid[x,y,0] = True
            if x > 0 and not occupancy_grid[x-1,y]:
                border_grid[x,y] = True
                dir_grid[x,y,1] = True
            if y < self.N_grid_y - 1 and not occupancy_grid[x,y+1]:
                border_grid[x,y] = True
                dir_grid[x,y,2] = True
            if x < self.N_grid_x - 1 and not occupancy_grid[x+1,y]:  # iterative through right, up, left, down
                border_grid[x,y] = True
                dir_grid[x,y,3] = True
        
        x_border, y_border = np.where(border_grid)
        dirs = dir_grid[x_border, y_border]
        
        # Debug draws:
        if self.debug_show_blob_shapes:
            vsk.stroke(2)
            for xy, r in zip(xy_metaballs, r_metaballs):
                vsk.circle(xy[0], xy[1], radius=r)
            
        if self.debug_show_blob:
            vsk.stroke(3)
            for i, x in enumerate(self.xs_grid):
                for j, y in enumerate(self.ys_grid):
                    vsk.circle(x, y, radius=0.01)
                    if occupancy_grid[i,j]:
                        vsk.rect(x, y, self.unit_size, self.unit_size, mode="center")
        
        if self.debug_show_blob_border:
            vsk.stroke(4)
            for i, x in enumerate(self.xs_grid):
                for j, y in enumerate(self.ys_grid):
                    if border_grid[i,j]:
                        vsk.rect(x, y, self.unit_size, self.unit_size, mode="center")
                        if dir_grid[i,j,0]:
                            vsk.line(x, y - 0.5*self.unit_size, x, y - self.unit_size)
                        if dir_grid[i,j,1]:
                            vsk.line(x - 0.5*self.unit_size, y, x - self.unit_size, y)
                        if dir_grid[i,j,2]:
                            vsk.line(x, y + 0.5*self.unit_size, x, y + self.unit_size)
                        if dir_grid[i,j,3]:
                            vsk.line(x + 0.5*self.unit_size, y, x + self.unit_size, y)
                            
        return x_valid, y_valid, x_border, y_border, dirs
    
    def draw(self, vsk: vsketch.Vsketch) -> None:
        self.draw_init(vsk)
        
        x_valid, y_valid, x_border, y_border, dirs = self.generate_occupancy_grid(vsk)

        # Draw shapes:
        vsk.stroke(4)
        for i in range(self.N_shapes):
            choice = pick_random_element(self.p_shapes)
            
            # Sample position and angle of shape:
            if self.shapes(choice + 1) in self.border_shapes:
                x, y, index = self.sample_random_occupied_pos(x_border, y_border, unit_dist=self.unit_size)
                dirs_i = dirs[index]
                dir_i = np.random.choice(np.where(dirs_i)[0])  # choose direction
                angle = self.rng.uniform(-0.25 * np.pi, 0.25 * np.pi) - 0.5 * np.pi * dir_i
            else:
                x, y, _ = self.sample_random_occupied_pos(x_valid, y_valid, unit_dist=self.unit_size)
                angle = self.rng.uniform(0, np.pi)
            
            # Determine size scaling of shape:
            if self.do_size_scaling:
                s = (self.N_shapes - 1 - i) / (self.N_shapes - 1)
                size_scale = self.size_scale_start + s * (self.size_scale_end - self.size_scale_start)
            else:
                size_scale = 1.0    
                
            if choice == enum_type_to_int(self.shapes.SHADED_RECT):
                width = size_scale * self.rng.uniform(self.rect_width_min, self.rect_width_max)
                height = width * self.rng.uniform(1.0, self.rect_height_gain_max)
                fill_distance = self.rng.uniform(self.fill_distance_min, self.fill_distance_max)
                rect = draw_shaded_rect(0, 0, width, height, fill_distance)
                self.draw_sketch_with_angle(vsk, rect, x, y, angle)
            elif choice == enum_type_to_int(self.shapes.SHADED_CIRCLE):
                radius = size_scale * self.rng.uniform(self.circle_radius_min, self.circle_radius_max)
                fill_distance = self.rng.uniform(self.fill_distance_min, self.fill_distance_max)
                vsk.sketch(draw_shaded_circle(x, y, radius, fill_distance, angle))
            elif choice == enum_type_to_int(self.shapes.RECT):
                width = size_scale * self.rng.uniform(self.rect_width_min, self.rect_width_max)
                height = width * self.rng.uniform(1.0, self.rect_height_gain_max)
                rect = draw_rect(0, 0, width, height)
                self.draw_sketch_with_angle(vsk, rect, x, y, angle)
            elif choice == enum_type_to_int(self.shapes.CIRCLE):
                radius = size_scale * self.rng.uniform(self.circle_radius_min, self.circle_radius_max)
                vsk.sketch(draw_circle(x, y, radius))
            elif choice == enum_type_to_int(self.shapes.FILLED_RECT):
                width = size_scale * self.rng.uniform(self.rect_width_min, self.rect_width_max)
                height = width * self.rng.uniform(1.0, self.rect_height_gain_max)
                rect = draw_filled_rect(0, 0, width, height)
                self.draw_sketch_with_angle(vsk, rect, x, y, angle)
            elif choice == enum_type_to_int(self.shapes.FILLED_CIRCLE):
                radius = size_scale * self.rng.uniform(self.circle_radius_min, self.circle_radius_max)
                vsk.sketch(draw_filled_circle(x, y, radius))
            elif choice == enum_type_to_int(self.shapes.DOT_CIRCLE):
                radius = size_scale * self.rng.uniform(self.dot_circle_radius_min, self.dot_circle_radius_max)
                radius_inner = radius * self.rng.uniform(self.dot_circle_inner_radius_gain_min,
                                                         self.dot_circle_inner_radius_gain_max)
                vsk.sketch(draw_dot_circle(x, y, radius, radius_inner))
            elif choice == enum_type_to_int(self.shapes.DOT_SHADED_RECT):
                width = size_scale * self.rng.uniform(self.rect_width_min, self.rect_width_max)
                height = width * self.rng.uniform(1.0, self.rect_height_gain_max)
                dot_distance = self.rng.uniform(self.dot_fill_distance_min, self.dot_fill_distance_max)
                if self.rng.uniform() < self.p_noisy_dot_fill:
                    noise_gain = self.rng.uniform(self.dot_fill_noise_gain_min, self.dot_fill_noise_gain_max)
                    noise_freq = self.rng.uniform(self.dot_fill_noise_freq_min, self.dot_fill_noise_freq_max)
                    rect = draw_dot_shaded_rect(0, 0, width, height, dot_distance, vsk, noise_gain=noise_gain, noise_freq=noise_freq)
                else:
                    rect = draw_dot_shaded_rect(0, 0, width, height, dot_distance)
                self.draw_sketch_with_angle(vsk, rect, x, y, angle)
            elif choice == enum_type_to_int(self.shapes.DOT_SHADED_CIRCLE):
                radius = size_scale * self.rng.uniform(self.circle_radius_min, self.circle_radius_max)
                dot_distance = self.rng.uniform(self.dot_fill_distance_min, self.dot_fill_distance_max)
                if self.rng.uniform() < self.p_noisy_dot_fill:
                    noise_gain = self.rng.uniform(self.dot_fill_noise_gain_min, self.dot_fill_noise_gain_max)
                    noise_freq = self.rng.uniform(self.dot_fill_noise_freq_min, self.dot_fill_noise_freq_max)
                    circle = draw_dot_shaded_circle(0, 0, radius, dot_distance, vsk, noise_gain=noise_gain, noise_freq=noise_freq)
                else:
                    circle = draw_dot_shaded_circle(0, 0, radius, dot_distance)
                self.draw_sketch_with_angle(vsk, circle, x, y, angle)
            elif choice == enum_type_to_int(self.shapes.POLE):
                height = self.rng.uniform(self.pole_height_min, self.pole_height_max)
                width = height * self.rng.uniform(self.pole_width_gain_min, self.pole_width_gain_max)
                radius = width * self.rng.uniform(self.pole_radius_gain_min, self.pole_radius_gain_max)
                pole = draw_pole(0, 0, width, height, radius)
                self.draw_sketch_with_angle(vsk, pole, x, y, angle)
            elif choice == enum_type_to_int(self.shapes.FLAG):
                height = self.rng.uniform(self.pole_height_min, self.pole_height_max)
                width = height * self.rng.uniform(self.pole_width_gain_min, self.pole_width_gain_max)
                flag_height = height * self.rng.uniform(self.flag_height_gain_min, self.flag_height_gain_max)
                flag_width = flag_height * self.rng.uniform(self.flag_width_gain_min, self.flag_width_gain_max)
                right = self.rng.uniform() < 0.5
                triangular = self.rng.uniform() < self.p_flag_triangular
                flag = draw_flag(0, 0, width, height, flag_width, flag_height, right, triangular)
                self.draw_sketch_with_angle(vsk, flag, x, y, angle)
            elif choice == enum_type_to_int(self.shapes.LINE):
                length = self.rng.uniform(self.line_length_min, self.line_length_max)
                line = draw_line(0, 0, length)
                self.draw_sketch_with_angle(vsk, line, x, y, angle)
                
        if self.debug_show_shapes:
            self.draw_debug_shapes(vsk)
            
        if self.occult:
            vsk.vpype("occult -i")
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")

if __name__ == "__main__":
    SchlagSketch.display()
