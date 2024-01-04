import vsketch
import numpy as np
from numpy.random import default_rng
from enum import Enum
import shapely as sh
import random
from shapely import Polygon, MultiPolygon, Point, MultiPoint, LineString
from plotter_shapes.plotter_shapes import get_empty_sketch, draw_filled_rect, draw_shaded_rect
from plotter_util.plotter_util import pick_random_element


def normalize_vec_to_sum_one(vec):
    return vec / np.sum(vec)


def normalize_mat_to_row_sum_one(mat):
    return mat / np.sum(mat, axis=-1)[:,None]


def get_points_iterable(points):
    if points.is_empty:
        return []
    elif points.geom_type == "MultiPoint":
        return points.geoms
    elif points.geom_type == "Point":
        return [points]


class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    
    def __str__(self):
        return self.name


def direction_to_unit_vector(direction):
    if direction == Direction.RIGHT:
        return np.array([1.0, 0.0])
    elif direction == Direction.UP:
        return np.array([0.0, -1.0])
    elif direction == Direction.LEFT:
        return np.array([-1.0, 0.0])
    elif direction == Direction.DOWN:
        return np.array([0.0, 1.0])
    else:
        raise Exception("Wrong direction.")


def direction_to_angle(direction):
    if direction == Direction.RIGHT:
        return 0.0
    elif direction == Direction.UP:
        return 0.5 * np.pi
    elif direction == Direction.LEFT:
        return np.pi
    elif direction == Direction.DOWN:
        return 1.5 * np.pi
    else:
        raise Exception("Wrong direction.")
    

def directions_are_normal(dir_1, dir_2):
    return (dir_1 in (Direction.RIGHT, Direction.LEFT) and dir_2 in (Direction.UP, Direction.DOWN)) or \
           (dir_1 in (Direction.UP, Direction.DOWN) and dir_2 in (Direction.RIGHT, Direction.LEFT))


class Module:
    def __init__(self, x, y, width, height, direction, from_module, allow_all_dirs=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.direction = direction
        
        if self.direction in (Direction.RIGHT, Direction.LEFT):
            self.length_x, self.length_y = width, height
        else:
            self.length_x, self.length_y = height, width            
        
        dir_unit_vec = direction_to_unit_vector(self.direction)
        self.x_center, self.y_center = 0.5 * np.array([self.length_x, self.length_y]) * dir_unit_vec + \
                                       np.array([self.x, self.y])

        self.allow_all_dirs = allow_all_dirs
        self.open_points = self.init_open_points(allow_all_dirs=self.allow_all_dirs)
        
    @classmethod
    def sample_bb_dims(cls, rng, from_height, limit_height_by_from_height=False, match_from_height=False):
        """Default sampling of bounding box size in local coordinates"""
        if match_from_height:
            height = from_height
        else:
            height_max = np.min((cls.height_max, from_height)) if limit_height_by_from_height else cls.height_max
            if height_max < cls.height_min:
                height = height_max
            else:
                height = rng.uniform(cls.height_min, height_max)
        width = height * rng.uniform(cls.width_gain_min, cls.width_gain_max)
        return width, height
    
    @classmethod
    def update(cls, height_min, height_max, width_gain_min, width_gain_max):
        """Default update of class variables"""
        cls.height_min = height_min
        cls.height_max = height_max
        cls.width_gain_min = width_gain_min
        cls.width_gain_max = width_gain_max
    
    def get_bounding_box(self, shrink=0.0):
        assert shrink < 0.5 * min((self.length_x, self.length_y))
        return sh.box(self.x_center - 0.5 * self.length_x + shrink, self.y_center - 0.5 * self.length_y + shrink,
                      self.x_center + 0.5 * self.length_x - shrink, self.y_center + 0.5 * self.length_y - shrink)
    
    def init_open_points(self, dist=2e-1, allow_all_dirs=False):
        sides = {}
        if self.direction != Direction.RIGHT or allow_all_dirs:
            sides[Direction.LEFT] = MultiPoint([(self.x_center - 0.5 * self.length_x, self.y_center)])
        if self.direction != Direction.UP or allow_all_dirs:
            sides[Direction.DOWN] = MultiPoint([(self.x_center, self.y_center + 0.5 * self.length_y)])
        if self.direction != Direction.LEFT or allow_all_dirs:
            sides[Direction.RIGHT] = MultiPoint([(self.x_center + 0.5 * self.length_x, self.y_center)])
        if self.direction != Direction.DOWN or allow_all_dirs:
            sides[Direction.UP] = MultiPoint([(self.x_center, self.y_center - 0.5 * self.length_y)])
        
        # TODO: add margin on ends of linspaces, we dont want to draw stuff all the way to the end...
        # if self.direction in (Direction.RIGHT, Direction.LEFT):
        #     n_points_x = int(np.round(self.width / dist))
        #     points_x = 0.5 * np.linspace(-self.width, self.width, num=n_points_x, endpoint=True) + self.x_center
            
        #     points_up = [(x, y) for (x, y) in zip(points_x, n_points_x * [self.y_center - 0.5 * self.height])]
        #     sides[Direction.UP] = MultiPoint(points_up)
            
        #     points_down = [(x, y) for (x, y) in zip(points_x, n_points_x * [self.y_center + 0.5 * self.height])]
        #     sides[Direction.DOWN] = MultiPoint(points_down)
            
        #     if self.direction == Direction.RIGHT:
        #         sides[Direction.RIGHT] = MultiPoint([(self.x_center + 0.5 * self.width, self.y_center)])
        #     else:
        #         sides[Direction.LEFT] = MultiPoint([(self.x_center - 0.5 * self.width, self.y_center)])
        # else:  # direction in UP, DOWN
        #     n_points_y = int(np.round(self.height / dist))
        #     points_y = 0.5 * np.linspace(-self.height, self.height, num=n_points_y, endpoint=True) + self.y_center
            
        #     points_right = [(x, y) for (x, y) in zip(n_points_y * [self.x_center + 0.5 * self.width], points_y)]
        #     sides[Direction.RIGHT] = MultiPoint(points_right)
            
        #     points_left = [(x, y) for (x, y) in zip(n_points_y * [self.x_center - 0.5 * self.width], points_y)]
        #     sides[Direction.LEFT] = MultiPoint(points_left)
            
        #     if self.direction == Direction.UP:
        #         sides[Direction.UP] = MultiPoint([(self.x_center, self.y_center - 0.5 * self.height)])
        #     else:
        #         sides[Direction.DOWN] = MultiPoint([(self.x_center, self.y_center + 0.5 * self.height)])
        return sides
    
    def get_sides(self):
        return list(self.open_points.keys())
    
    def get_edge_line(self):
        if self.direction in (Direction.RIGHT, Direction.LEFT):
            return LineString([[self.x, self.y - 0.5 * self.length_y], 
                               [self.x, self.y + 0.5 * self.length_y]])
        else:
            return LineString([[self.x - 0.5 * self.length_x, self.y], 
                               [self.x + 0.5 * self.length_x, self.y]])
    
    def draw_bounding_box(self):
        sketch = get_empty_sketch()
        sketch.stroke(2)
        sketch.rect(self.x_center, self.y_center, self.length_x, self.length_y, mode="center")
        sketch.circle(self.x, self.y, radius=3e-2)
        sketch.circle(self.x_center, self.y_center, radius=3e-2)
        return sketch
    
    def draw(self):
        sketch = get_empty_sketch()
        sketch.rect(self.x_center, self.y_center, self.length_x, self.length_y, mode="center")
        return sketch
    
    def init_sketch(self, center=False):
        sketch = get_empty_sketch()
        if center:
            sketch.translate(self.x_center, self.y_center)
        else:
            sketch.translate(self.x, self.y)
        sketch.rotate(-direction_to_angle(self.direction))
        return sketch
    

class Capsule(Module):
    def __init__(self, x, y, width, height, direction, from_module, allow_all_dirs=False):
        super().__init__(x, y, width, height, direction, from_module=from_module, allow_all_dirs=allow_all_dirs)

    @classmethod
    def sample_bb_dims(cls, rng, from_height, match_from_height=False):
        return super(Capsule, cls).sample_bb_dims(rng, from_height, match_from_height=match_from_height)
    
    
class CapsuleVariation1(Capsule):
    def draw(self):
        sketch = super().draw()
        return sketch


class CapsuleMultiWindow(Capsule):
    window_types = ["CIRCLE", "SQUARE", "SQUARE_ROUNDED"]
    line_types = ["EMPTY", "PARALLEL", "NORMAL", "BOX"]

    def __init__(self, x, y, width, height, direction, from_module, allow_all_dirs=False):
        super().__init__(x, y, width, height, direction, from_module, allow_all_dirs=allow_all_dirs)
        self.window_size = self.height * np.random.uniform(self.window_size_gain_min, self.window_size_gain_max)
        self.windows_size = self.width * np.random.uniform(self.windows_size_gain_min, self.windows_size_gain_max)
        self.nonwindows_size = (self.width - self.windows_size)
        self.window_dist = self.window_size * self.window_dist_gain
        
        self.num_windows = int((self.windows_size + self.window_dist) / (self.window_size + self.window_dist))
        if self.num_windows > 1:
            self.window_dist = (self.window_dist - self.num_windows * self.window_size)  / (self.num_windows - 1)  # augment the "between window distance" to match the rounding to get integer number of  windows
        else:
            self.window_dist = 0.0
            
        self.window_rounded_radius = self.window_size * np.random.uniform(self.window_rounded_radius_gain_min, self.window_rounded_radius_gain_max)

        self.parallel_line_dist = self.height * np.random.uniform(self.parallel_line_dist_gain_min, self.parallel_line_dist_gain_max)
        
    @classmethod
    def update(cls, window_probs, window_size_gain_min, window_size_gain_max, windows_size_gain_min, windows_size_gain_max, 
               window_dist_gain_min, window_dist_gain_max, window_rounded_radius_gain_min, window_rounded_radius_gain_max, line_probs, parallel_line_dist_gain_min, 
               parallel_line_dist_gain_max):
        cls.window_choice = pick_random_element(cls.window_types, window_probs)
        cls.line_choice = pick_random_element(cls.line_types, line_probs)
        cls.window_size_gain_min = window_size_gain_min
        cls.window_size_gain_max = window_size_gain_max
        cls.windows_size_gain_min = windows_size_gain_min
        cls.windows_size_gain_max = windows_size_gain_max
        cls.window_dist_gain = np.random.uniform(window_dist_gain_min, window_dist_gain_max)
        cls.window_rounded_radius_gain_min = window_rounded_radius_gain_min
        cls.window_rounded_radius_gain_max = window_rounded_radius_gain_max
        cls.line_probs = line_probs
        cls.parallel_line_dist_gain_min = parallel_line_dist_gain_min
        cls.parallel_line_dist_gain_max = parallel_line_dist_gain_max
        
    def draw(self):
        sketch = self.init_sketch(center=True)
        sketch.rect(0, 0, self.width, self.height, mode="center")

        if self.num_windows > 1:
            xs = np.linspace(-0.5 * (self.windows_size - self.window_size), 0.5 * (self.windows_size - self.window_size), num=self.num_windows)
        else:
            xs = [0.0]
        for x in xs:
            if self.window_choice == "CIRCLE":
                sketch.circle(x, 0, self.window_size)
            elif self.window_choice == "SQUARE":
                sketch.square(x, 0, self.window_size, mode="center")
            elif self.window_choice == "SQUARE_ROUNDED":
                sketch.rect(x, 0, self.window_size, self.window_size, self.window_rounded_radius, mode="center")
        
        if self.line_choice in ("PARALLEL", "NORMAL", "BOX"):
            line_y = 0.5 * self.height - self.parallel_line_dist
            line_x = 0.5 * self.width - 0.25 * self.nonwindows_size
            if self.line_choice == "PARALLEL":
                sketch.line(-0.5 * self.width, line_y, 0.5 * self.width, line_y)
                sketch.line(-0.5 * self.width, -line_y, 0.5 * self.width, -line_y)
            elif self.line_choice == "NORMAL":
                sketch.line(-line_x, -0.5 * self.height, -line_x, 0.5 * self.height)
                sketch.line(line_x, -0.5 * self.height, line_x, 0.5 * self.height)
            elif self.line_choice == "BOX":
                sketch.rect(0, 0, 2*line_x, 2*line_y, mode="center")
        return sketch
    

class Capsule3D(Capsule):
    def __init__(self, x, y, width, height, direction, from_module, allow_all_dirs=False):
        super().__init__(x, y, width, height, direction, from_module, allow_all_dirs=allow_all_dirs)
        self.num_lines = np.random.randint(self.num_lines_min, self.num_lines_max + 1)
        self.sin_stop = 0.45
        
    @classmethod
    def update(cls, num_lines_min, num_lines_max):
        cls.num_lines_min = num_lines_min
        cls.num_lines_max = num_lines_max
        
    def draw(self):
        sketch = self.init_sketch(center=True)
        sketch.rect(0, 0, self.width, self.height, mode="center")
        
        ys = 0.5 * self.height * np.sin(np.pi * np.linspace(-self.sin_stop, self.sin_stop, num=self.num_lines))
        for y in ys:
            sketch.line(-0.5 * self.width, y, 0.5 * self.width, y)
            
        return sketch
    
    
class CapsuleParallelLines(Capsule):
    def __init__(self, x, y, width, height, direction, from_module, allow_all_dirs=False):
        super().__init__(x, y, width, height, direction, from_module, allow_all_dirs=allow_all_dirs)
        self.num_lines = np.random.randint(self.num_lines_min, self.num_lines_max + 1)
        
    @classmethod
    def update(cls, num_lines_min, num_lines_max):
        cls.num_lines_min = num_lines_min
        cls.num_lines_max = num_lines_max
        
    def draw(self):
        sketch = self.init_sketch(center=True)
        
        sketch.rect(0, 0, self.width, self.height, mode="center")
        
        line_dist = self.height / (self.num_lines + 1)
        height_offseted = 0.5 * self.height - line_dist
        ys = np.linspace(-height_offseted, height_offseted, num=self.num_lines)
        for y in ys:
            sketch.line(-0.5 * self.width, y, 0.5 * self.width, y)
            
        return sketch


class CapsuleNormalLines(Capsule):
    line_types = ["RANDOM", "DOUBLE_THIN", "DOUBLE_FLAT", "DOUBLE_MULTI", "DOUBLE_MULTI_RANDOM", "DOUBLE_SHADED", "DOUBLE_BLACK"]

    def __init__(self, x, y, width, height, direction, from_module, allow_all_dirs=False):
        super().__init__(x, y, width, height, direction, from_module, allow_all_dirs=allow_all_dirs)
        if self.line_choice == "RANDOM":
            self.num_lines_rand = np.random.randint(self.num_lines_rand_min, self.num_lines_rand_max + 1)
        else:
            self.double_offset = 0.5 * self.width * np.random.uniform(self.double_offset_gain_min, self.double_offset_gain_max)
            self.double_dist = 0.5 * self.width * np.random.uniform(self.double_dist_gain_min, self.double_dist_gain_max)
            if self.line_choice in ("DOUBLE_MULTI", "DOUBLE_MULTI_RANDOM"):
                self.num_lines_multi = np.random.randint(self.num_lines_multi_min, self.num_lines_multi_max + 1)
                self.double_multi_dist = 0.5 * self.width * np.random.uniform(self.double_multi_dist_gain_min, self.double_multi_dist_gain_max)
            elif self.line_choice == "DOUBLE_SHADED":
                self.double_shaded_dist = np.random.uniform(self.double_shaded_dist_min, self.double_shaded_dist_max)
        
    @classmethod
    def update(cls, probs, num_lines_rand_min, num_lines_rand_max, double_offset_gain_min, double_offset_gain_max, double_dist_gain_min, 
               double_dist_gain_max, num_lines_multi_min, num_lines_multi_max, double_multi_dist_gain_min, double_multi_dist_gain_max,
               double_shaded_dist_min, double_shaded_dist_max):
        cls.line_choice = pick_random_element(cls.line_types, probs)
        cls.num_lines_rand_min = num_lines_rand_min
        cls.num_lines_rand_max = num_lines_rand_max
        cls.double_offset_gain_min = double_offset_gain_min
        cls.double_offset_gain_max = double_offset_gain_max
        cls.double_dist_gain_min = double_dist_gain_min
        cls.double_dist_gain_max = double_dist_gain_max
        cls.num_lines_multi_min = num_lines_multi_min
        cls.num_lines_multi_max = num_lines_multi_max
        cls.double_multi_dist_gain_min = double_multi_dist_gain_min
        cls.double_multi_dist_gain_max = double_multi_dist_gain_max
        cls.double_shaded_dist_min = double_shaded_dist_min
        cls.double_shaded_dist_max = double_shaded_dist_max
        
    def draw(self):
        sketch = self.init_sketch(center=True)
        sketch.rect(0, 0, self.width, self.height, mode="center")
        
        if self.line_choice == "RANDOM":
            for x in np.random.uniform(-0.5 * self.width, 0.5 * self.width, size=self.num_lines_rand):
                sketch.line(x, -0.5 * self.height, x, 0.5 * self.height)
        else:
            line_pos = 0.5 * self.width - self.double_offset
            if self.line_choice == "DOUBLE_THIN":
                sketch.line(-line_pos, -0.5 * self.height, -line_pos, 0.5 * self.height)
                sketch.line(line_pos, -0.5 * self.height, line_pos, 0.5 * self.height)
            elif self.line_choice == "DOUBLE_FLAT":
                line_pos_1 = line_pos + 0.5 * self.double_dist
                line_pos_2 = line_pos - 0.5 * self.double_dist
                sketch.line(-line_pos_1, -0.5 * self.height, -line_pos_1, 0.5 * self.height)
                sketch.line(-line_pos_2, -0.5 * self.height, -line_pos_2, 0.5 * self.height)
                sketch.line(line_pos_1, -0.5 * self.height, line_pos_1, 0.5 * self.height)
                sketch.line(line_pos_2, -0.5 * self.height, line_pos_2, 0.5 * self.height)
            if self.line_choice in ("DOUBLE_MULTI", "DOUBLE_MULTI_RANDOM"):
                line_pos_1 = line_pos + 0.5 * self.double_multi_dist
                line_pos_2 = line_pos - 0.5 * self.double_multi_dist
                if self.line_choice == "DOUBLE_MULTI":
                    for x in np.linspace(line_pos_1, line_pos_2, num=self.num_lines_multi):
                        sketch.line(-x, -0.5 * self.height, -x, 0.5 * self.height)
                        sketch.line(x, -0.5 * self.height, x, 0.5 * self.height)
                else:
                    for x in np.random.uniform(line_pos_1, line_pos_2, size=self.num_lines_multi):
                        sketch.line(-x, -0.5 * self.height, -x, 0.5 * self.height)
                        sketch.line(x, -0.5 * self.height, x, 0.5 * self.height)
            elif self.line_choice == "DOUBLE_SHADED":
                sketch.sketch(draw_shaded_rect(line_pos, 0, self.double_dist, self.height, fill_distance=self.double_shaded_dist))
                sketch.sketch(draw_shaded_rect(-line_pos, 0, self.double_dist, self.height, fill_distance=self.double_shaded_dist))
            elif self.line_choice == "DOUBLE_BLACK":
                sketch.sketch(draw_filled_rect(line_pos, 0, self.double_dist, self.height))
                sketch.sketch(draw_filled_rect(-line_pos, 0, self.double_dist, self.height))
        return sketch
    

class SquareCapsule(Capsule):
    def __init__(self, x, y, width, height, direction, from_module, allow_all_dirs=False):
        super().__init__(x, y, width, height, direction, from_module, allow_all_dirs=allow_all_dirs)
        self.corner_radius = 0.5 * self.width * np.random.uniform(self.corner_radius_gain_min, self.corner_radius_gain_max)
        self.border_size = self.width * np.random.uniform(self.border_gain_min, self.border_gain_max)
        self.outer_circle_radius = 0.5 * self.width * np.random.uniform(self.outer_circle_gain_min, self.outer_circle_gain_max)
        self.inner_circle_radius = self.outer_circle_radius * np.random.uniform(self.inner_circle_gain_min, self.inner_circle_gain_max)
        
    @classmethod
    def update(cls, height_min, height_max, border_prob, cross_prob, shaded_circle_prob, rounded_corners_prob, corner_radius_gain_min, corner_radius_gain_max,
               border_gain_min, border_gain_max, outer_circle_gain_min, outer_circle_gain_max, inner_circle_gain_min,
               inner_circle_gain_max, num_lines_shaded_circle_min, num_lines_shaded_circle_max):
        cls.draw_border = np.random.rand() < border_prob
        cls.draw_cross = np.random.rand() < cross_prob
        cls.draw_shaded_circle = np.random.rand() < shaded_circle_prob
        cls.draw_rounded_corners = np.random.rand() < rounded_corners_prob
        cls.height_min = height_min
        cls.height_max = height_max
        cls.width_gain_min, cls.width_gain_max = 1.0, 1.0
        cls.corner_radius_gain_min = corner_radius_gain_min
        cls.corner_radius_gain_max = corner_radius_gain_max
        cls.border_gain_min = border_gain_min
        cls.border_gain_max = border_gain_max
        cls.outer_circle_gain_min = outer_circle_gain_min
        cls.outer_circle_gain_max = outer_circle_gain_max
        cls.inner_circle_gain_min = inner_circle_gain_min
        cls.inner_circle_gain_max = inner_circle_gain_max
        cls.num_lines_shaded_circle = 2 * int(0.5 * np.random.randint(num_lines_shaded_circle_min, num_lines_shaded_circle_max))
        
    def draw(self):
        sketch = get_empty_sketch()
        sketch.translate(self.x_center, self.y_center)
        
        if self.draw_rounded_corners:
            sketch.rect(0, 0, self.height, self.width, self.corner_radius, mode="center")
            if self.draw_border:
                sketch.rect(0, 0, self.border_size, self.border_size, 
                            self.corner_radius * self.border_size / self.width, mode="center")
        else:
            sketch.rect(0, 0, self.height, self.width, mode="center")
            if self.draw_border:
                sketch.rect(0, 0, self.border_size, self.border_size, mode="center")
        
        sketch.circle(0, 0, radius=self.outer_circle_radius)
        sketch.circle(0, 0, radius=self.inner_circle_radius)
        
        if self.draw_shaded_circle:
            with sketch.pushMatrix():
                theta = 2 * np.pi / self.num_lines_shaded_circle
                for i in range(self.num_lines_shaded_circle):
                    sketch.line(0, self.outer_circle_radius, 0, self.inner_circle_radius)
                    sketch.rotate(theta)
        
        if self.draw_cross:
            with sketch.pushMatrix():
                sketch.rotate(0.25 * np.pi)
                sketch.line(0, self.inner_circle_radius, 0, -self.inner_circle_radius)
                sketch.line(self.inner_circle_radius, 0, -self.inner_circle_radius, 0)

        return sketch    


class Connector(Module):
    def __init__(self, x, y, width, height, direction, from_module):
        # Determine end-module height (for drawing), and update height accordingly to get correct BB size:
        dirs_normal = directions_are_normal(direction, from_module.direction)
        self.from_height = from_module.width if dirs_normal else from_module.height
        if dirs_normal:
            end_height = np.random.uniform(Connector.height_min, Connector.height_max)
        else:
            end_height = self.from_height * np.random.uniform(Connector.from_height_gain_min, Connector.from_height_gain_max)
        self.end_height = np.clip(end_height, Connector.height_min, Connector.height_max)
        self.start_height = height
        height = np.max((height, self.end_height))
        
        super().__init__(x, y, width, height, direction, from_module)
        self.open_points = dict(zip([self.direction], [self.open_points[self.direction]]))  # connector type can only build forward
        
    @classmethod
    def update(cls, height_min, height_max, from_height_gain_min, from_height_gain_max, width_gain_min, width_gain_max):
        """Default update of class variables"""
        cls.height_min = height_min
        cls.height_max = height_max
        cls.from_height_gain_min = from_height_gain_min
        cls.from_height_gain_max = from_height_gain_max
        cls.width_gain_min = width_gain_min
        cls.width_gain_max = width_gain_max
    
    
class ConnectorTrapezoid(Connector):
    def draw(self):
        sketch = self.init_sketch(center=True)
        sketch.polygon([(-0.5 * self.width, -0.5 * self.start_height), 
                        (0.5 * self.width, -0.5 * self.end_height), 
                        (0.5 * self.width, 0.5 * self.end_height),
                        (-0.5 * self.width, 0.5 * self.start_height)])
        return sketch


class ConnectorTrapezoidExtended(Connector):
    connector_types = ["UNION", "EMPTY", "SHADED"]
    
    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)
        self.height = np.min((self.start_height, self.end_height)) * np.random.uniform(self.height_gain_min, self.height_gain_max)
        self.flat_width = self.width * self.flat_width_gain
        self.trapezoid_width = self.width - self.flat_width
        
    @classmethod
    def update(cls, height_min, height_max, from_height_gain_min, from_height_gain_max, width_gain_min, width_gain_max,
               height_gain_min, height_gain_max, flat_width_gain_min, flat_width_gain_max, shaded_dist_min, shaded_dist_max, probs):
        super(ConnectorTrapezoidExtended, cls).update(height_min, height_max, from_height_gain_min, from_height_gain_max, 
                                           width_gain_min, width_gain_max)
        cls.connector_choice = pick_random_element(cls.connector_types, probs)
        cls.height_gain_min = height_gain_min
        cls.height_gain_max = height_gain_max
        cls.flat_width_gain = np.random.uniform(flat_width_gain_min, flat_width_gain_max)
        cls.shaded_dist = np.random.uniform(shaded_dist_min, shaded_dist_max)
        
    def draw(self):
        sketch = self.init_sketch()
        # 50% chance to rotate
        
        if self.connector_choice == "UNION":
            sketch.polygon([(0, -0.5 * self.start_height), 
                            (self.flat_width, -0.5 * self.height), 
                            (self.width, -0.5 * self.height), 
                            (self.width, 0.5 * self.height), 
                            (self.flat_width, 0.5 * self.height),
                            (0, 0.5 * self.start_height)])
        elif self.connector_choice in ("EMPTY", "SHADED"):
            sketch.polygon([(0, -0.5 * self.start_height), 
                            (self.flat_width, -0.5 * self.height), 
                            (self.flat_width, 0.5 * self.height),
                            (0, 0.5 * self.start_height)])
            if self.connector_choice in "EMPTY":
                sketch.rect(self.flat_width + 0.5 * self.trapezoid_width, 0, self.trapezoid_width, self.height, mode="center")
            elif self.connector_choice in "SHADED":
                sketch.sketch(draw_shaded_rect(self.flat_width + 0.5 * self.trapezoid_width, 0, self.trapezoid_width, self.height,
                                               fill_distance=self.shaded_dist))
        
        return sketch
    
    
class ConnectorSimple(Connector):
    connector_types = ["EMPTY", "SHADED", "FILLED"]

    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)
        self.height = np.min((self.start_height, self.end_height)) * np.random.uniform(self.height_gain_min, self.height_gain_max)

    @classmethod
    def update(cls, height_min, height_max, from_height_gain_min, from_height_gain_max, width_gain_min, width_gain_max, 
               height_gain_min, height_gain_max, shaded_dist_min, shaded_dist_max, probs):
        super(ConnectorSimple, cls).update(height_min, height_max, from_height_gain_min, from_height_gain_max, width_gain_min, width_gain_max)
        cls.connector_choice = pick_random_element(cls.connector_types, probs)
        cls.height_gain_min = height_gain_min
        cls.height_gain_max = height_gain_max
        cls.shaded_dist = np.random.uniform(shaded_dist_min, shaded_dist_max)
        
    def draw(self):
        sketch = self.init_sketch(center=True)
        if self.connector_choice == "EMPTY":
            sketch.rect(0, 0, self.width, self.height, mode="center")
        elif self.connector_choice == "SHADED":
            sketch.sketch(draw_shaded_rect(0, 0, self.width, self.height, fill_distance=self.shaded_dist))
        elif self.connector_choice == "FILLED":
            sketch.sketch(draw_filled_rect(0, 0, self.width, self.height))
        return sketch
        
class SolarPanel(Module):
    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)
        self.open_points = None  # dont build outwards from this module
                
        self.num_panels_x = int(np.round(self.width / self.panel_dist_approx))
        self.panel_dist = self.width / self.num_panels_x
        
    @classmethod
    def update(cls, height_min, height_max, width_gain_min, width_gain_max, panel_num_y_min, panel_num_y_max, 
               panel_dist_x_min, panel_dist_x_max):
        super(SolarPanel, cls).update(height_min, height_max, width_gain_min, width_gain_max)
        cls.num_panels_y = np.random.randint(panel_num_y_min, panel_num_y_max + 1)
        cls.panel_dist_approx = np.random.uniform(panel_dist_x_min, panel_dist_x_max)


class SolarPanelSingle(SolarPanel):
    def draw(self):
        sketch = self.init_sketch()
        
        for y in np.linspace(-0.5 * self.height, 0.5 * self.height, self.num_panels_y + 1):
            sketch.line(0, y, self.width, y)
            
        for x in np.linspace(0, self.width, self.num_panels_x + 1):
            sketch.line(x, -0.5 * self.height, x, 0.5 * self.height)
        return sketch
    

class SolarPanelDouble(SolarPanel):
    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)
        self.panel_dist = np.random.uniform(self.panel_dist_min, self.panel_dist_max)
        self.panel_inset = np.random.uniform(self.panel_inset_min, self.panel_inset_max)
        
        if self.draw_multi_beams:
            self.width_per_beam = self.width / (self.n_beams_extra + 1)
            self.num_panels_x = int(np.round((self.width_per_beam) / self.panel_dist_approx))
            self.panel_dist = self.width_per_beam / self.num_panels_x
        
    @classmethod
    def update(cls, connector_width_min, connector_width_max, connector_height_min, 
               connector_height_max, panel_dist_min, panel_dist_max, panel_inset_min, panel_inset_max,
               multi_beams_prob, n_beams_extra_min, n_beams_extra_max):
        cls.connector_width = np.random.uniform(connector_width_min, connector_width_max)
        cls.connector_height = np.random.uniform(connector_height_min, connector_height_max)
        cls.panel_dist_min = panel_dist_min
        cls.panel_dist_max = panel_dist_max
        cls.panel_inset_min = panel_inset_min
        cls.panel_inset_max = panel_inset_max
        cls.draw_multi_beams = np.random.rand() < multi_beams_prob
        cls.n_beams_extra = np.random.randint(n_beams_extra_min, n_beams_extra_max + 1)
        
    def draw(self):
        sketch = self.init_sketch()

        # Connector:
        sketch.rect(0.5 * self.connector_width, 0, self.connector_width, self.connector_height, mode="center")
        sketch.translate(self.connector_width, 0)
        
        # The inner beam:
        if not self.draw_multi_beams:
            sketch.rect(0.5 * self.connector_height, 0, self.connector_height, self.height, mode="center")
        sketch.translate(self.connector_height, 0)
        
        # The panels:
        if self.draw_multi_beams:
            panel_width = self.width - self.connector_width - self.connector_height
            beam_dist = panel_width / (self.n_beams_extra + 1)
            for x_beam in np.linspace(0, panel_width - beam_dist, self.n_beams_extra + 1):
                for y in np.linspace(0.5 * self.panel_dist, 0.5 * self.height - self.panel_inset, self.num_panels_y + 1):
                    sketch.line(x_beam, -y, x_beam + beam_dist - self.connector_height, -y)
                    sketch.line(x_beam, y, x_beam + beam_dist - self.connector_height, y)
                
                for x in np.linspace(0, beam_dist - self.connector_height, self.num_panels_x + 1):
                    sketch.line(x_beam + x, 0.5 * self.panel_dist, x_beam + x, 0.5 * self.height - self.panel_inset)
                    sketch.line(x_beam + x, -0.5 * self.height + self.panel_inset, x_beam + x, -0.5 * self.panel_dist)
                
                sketch.rect(x_beam - 0.5 * self.connector_height, 0, self.connector_height, self.height, mode="center")
            sketch.translate(panel_width - self.connector_height, 0)
        else:
            panel_width = self.width - self.connector_width - 2 * self.connector_height
            for y in np.linspace(0.5 * self.panel_dist, 0.5 * self.height - self.panel_inset, self.num_panels_y + 1):
                sketch.line(0, -y, panel_width, -y)
                sketch.line(0, y, panel_width, y)
            
            for x in np.linspace(0, panel_width, self.num_panels_x + 1):
                sketch.line(x, 0.5 * self.panel_dist, x, 0.5 * self.height - self.panel_inset)
                sketch.line(x, -0.5 * self.height + self.panel_inset, x, -0.5 * self.panel_dist)
            sketch.translate(panel_width, 0)
        
        sketch.rect(0.5 * self.connector_height, 0, self.connector_height, self.height, mode="center")  # outer beam
        return sketch
    

class Decoration(Module):
    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)
        self.open_points = None  # dont build outwards from this module
    
    @classmethod
    def sample_bb_dims(cls, rng, from_height):
        return super(Decoration, cls).sample_bb_dims(rng, from_height, limit_height_by_from_height=True)
    

class Antenna(Decoration):
    antenna_types = ["EMPTY", "DOT", "SQUARE"]
    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)
        self.dot_radius = np.random.uniform(self.dot_radius_min, self.dot_radius_max)
        
    @classmethod
    def update(cls, height_min, height_max, width_gain_min, width_gain_max, probs, dot_radius_min, 
               dot_radius_max):
        super(Antenna, cls).update(height_min, height_max, width_gain_min, width_gain_max)
        cls.antenna_choice = pick_random_element(cls.antenna_types, probs)
        cls.dot_radius_min = dot_radius_min
        cls.dot_radius_max = dot_radius_max

    def draw(self):
        sketch = self.init_sketch()
        if self.antenna_choice in ("DOT", "SQUARE"):
            sketch.line(0, 0, self.width - 2 * self.dot_radius, 0)
            if self.antenna_choice == "DOT":
                sketch.circle(self.width - self.dot_radius, 0, radius=self.dot_radius)
            else:
                sketch.square(self.width - self.dot_radius, 0, self.dot_radius, mode="radius")
        elif self.antenna_choice == "EMPTY":
            sketch.line(0, 0, self.width, 0)
        return sketch


class DockingBaySimple(Decoration):
    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)

    @classmethod
    def update(cls, height_min, height_max, width_gain_min, width_gain_max, black_prob, shaded_dist_min,
               shaded_dist_max):
        super(DockingBaySimple, cls).update(height_min, height_max, width_gain_min, width_gain_max)
        cls.draw_black = np.random.rand() < black_prob
        cls.shaded_dist = np.random.uniform(shaded_dist_min, shaded_dist_max)
        
    def draw(self):
        sketch = self.init_sketch()
        if self.draw_black:
            sketch.sketch(draw_filled_rect(0.5 * self.width, 0, self.width, self.height))
        else:  # shaded
            sketch.sketch(draw_shaded_rect(0.5 * self.width, 0, self.width, self.height, fill_distance=self.shaded_dist))
        return sketch
    
    
class DockingBay(Decoration):
    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)
        self.flat_end_height = self.height * np.random.uniform(self.flat_end_height_gain_min, self.flat_end_height_gain_max)
        self.end_height = self.flat_end_height * np.random.uniform(self.end_height_gain_min, self.end_height_gain_max)
        
    @classmethod
    def update(cls, height_min, height_max, width_gain_min, width_gain_max, union_middle_prob, 
               flat_end_height_gain_min, flat_end_height_gain_max, end_height_gain_min, end_height_gain_max, 
               start_frac_min, start_frac_max, end_frac_min, end_frac_max, 
               flat_start_frac_min, flat_start_frac_max, flat_end_frac_min, flat_end_frac_max):
        super(DockingBay, cls).update(height_min, height_max, width_gain_min, width_gain_max)
        cls.union_middle = np.random.rand() < union_middle_prob
        cls.flat_end_height_gain_min = flat_end_height_gain_min
        cls.flat_end_height_gain_max = flat_end_height_gain_max
        cls.end_height_gain_min = end_height_gain_min
        cls.end_height_gain_max = end_height_gain_max
        cls.start_frac = np.random.uniform(start_frac_min, start_frac_max)
        cls.end_frac = np.random.uniform(end_frac_min, end_frac_max)
        cls.flat_start_frac = np.random.uniform(flat_start_frac_min, flat_start_frac_max)
        cls.flat_end_frac = np.random.uniform(flat_end_frac_min, flat_end_frac_max)
        
    def draw(self):
        sketch = self.init_sketch()
        
        sketch.sketch(draw_filled_rect(0.5 * self.width * self.start_frac, 0, self.width * self.start_frac, self.height))
        sketch.translate(self.width * self.start_frac, 0)
        
        width_middle = self.width * (1.0 - self.start_frac - self.end_frac)
        if self.union_middle:
            main_shape = sketch.createShape()
            main_shape.rect(0.5 * self.flat_start_frac * width_middle, 0, (self.flat_start_frac + 1e-3) * width_middle, self.height, mode="center")
            main_shape.polygon([(self.flat_start_frac * width_middle, 0.5 * self.height), 
                                ((1.0 - self.flat_end_frac) * width_middle, 0.5 * self.flat_end_height),
                                ((1.0 - self.flat_end_frac) * width_middle, -0.5 * self.flat_end_height),
                                (self.flat_start_frac * width_middle, -0.5 * self.height)], close=True)
            main_shape.rect((1.0 - 0.5 * self.flat_end_frac - 1e-3) * width_middle, 0, self.flat_end_frac * width_middle, self.flat_end_height, mode="center")
            sketch.shape(main_shape)
        else:
            sketch.rect(0.5 * self.flat_start_frac * width_middle, 0, (self.flat_start_frac + 1e-3) * width_middle, self.height, mode="center")
            sketch.polygon([(self.flat_start_frac * width_middle, 0.5 * self.height), 
                                ((1.0 - self.flat_end_frac) * width_middle, 0.5 * self.flat_end_height),
                                ((1.0 - self.flat_end_frac) * width_middle, -0.5 * self.flat_end_height),
                                (self.flat_start_frac * width_middle, -0.5 * self.height)], close=True)
            sketch.rect((1.0 - 0.5 * self.flat_end_frac - 1e-3) * width_middle, 0, self.flat_end_frac * width_middle, self.flat_end_height, mode="center")
        sketch.translate(width_middle, 0)
        
        sketch.sketch(draw_filled_rect(0.5 * self.end_frac * self.width, 0, self.width * self.end_frac, self.end_height))
        
        return sketch


class Boxes(Decoration):
    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)
        if self.draw_multi:
            self.num_box = np.random.randint(self.num_box_min, self.num_box_max + 1)
        else:
            self.num_box = 1
        self.draw_box_on_box = np.random.rand() < self.prob_box_on_box
        self.draw_line = np.random.rand() < self.prob_line_on_box
    
    @classmethod
    def update(cls, height_min, height_max, width_gain_min, width_gain_max, prob_multi, prob_box_on_box,
               prob_line_on_box, num_box_min, num_box_max, box_width_gain_min, box_width_gain_max,
               box_height_gain_min, box_height_gain_max, box_box_width_gain_min, box_box_width_gain_max, 
               box_box_height_gain_min, box_box_height_gain_max, line_length_gain_min, line_length_gain_max):
        super(Boxes, cls).update(height_min, height_max, width_gain_min, width_gain_max)
        cls.draw_multi = np.random.rand() < prob_multi
        cls.num_box_min = num_box_min
        cls.num_box_max = num_box_max
        cls.box_width_gain_min = box_width_gain_min
        cls.box_width_gain_max = box_width_gain_max
        cls.box_height_gain_min = box_height_gain_min
        cls.box_height_gain_max = box_height_gain_max
        cls.prob_box_on_box = prob_box_on_box
        cls.prob_line_on_box = prob_line_on_box
        cls.box_box_width_gain_min = box_box_width_gain_min
        cls.box_box_width_gain_max = box_box_width_gain_max
        cls.box_box_height_gain_min = box_box_height_gain_min
        cls.box_box_height_gain_max = box_box_height_gain_max
        cls.line_length_gain_min = line_length_gain_min
        cls.line_length_gain_max = line_length_gain_max
                
    def draw(self):
        sketch = self.init_sketch()
        
        box_width = self.width * np.random.uniform(self.box_width_gain_min, self.box_width_gain_max)
        box_height = box_width * np.random.uniform(self.box_height_gain_min, self.box_height_gain_max)
        box_y = np.random.uniform(-0.5 * (self.height - box_height), 0.5 * (self.height - box_height))
        sketch.rect(0.5 * box_height, box_y, box_height, box_width, mode="center")
        
        if self.draw_box_on_box:
            box_box_width = box_width * np.random.uniform(self.box_box_width_gain_min, self.box_box_width_gain_max)
            box_box_height = box_height * np.random.uniform(self.box_box_height_gain_min, self.box_box_height_gain_max)
            box_box_y = np.random.uniform(-0.5 * (box_height - box_box_height), 0.5 * (box_height - box_box_height))
            sketch.rect(box_height + 0.5 * box_box_height, box_y + box_box_y, box_box_height, box_box_width, mode="center")
        elif self.draw_line:
            line_length = box_width * np.random.uniform(self.line_length_gain_min, self.line_length_gain_max)
            line_y = np.random.uniform(-0.5 * box_width, 0.5 * box_width)
            sketch.line(box_height, box_y + line_y, box_height + line_length, box_y + line_y)
            
        return sketch
    

class Inflatable(Decoration):
    line_types = ["EMPTY", "PARALLEL", "NORMAL"]

    def __init__(self, x, y, width, height, direction, from_module):
        super().__init__(x, y, width, height, direction, from_module)        
        self.corner_radius = np.min((self.width, self.height)) * self.corner_radius_gain
        self.connector_height = self.height * np.random.uniform(self.dock_height_gain_min, self.dock_height_gain_max)
        self.connector_width = self.connector_height * np.random.uniform(self.dock_width_gain_min, self.dock_width_gain_max)
        self.shaded_connector_fill_dist = self.connector_height * self.shaded_dock_fill_dist_gain
        self.capsule_width = self.width - self.connector_width
        self.num_lines_parallel = np.random.randint(self.num_lines_parallel_min, self.num_lines_parallel_max + 1)
        self.num_lines_normal = np.random.randint(self.num_lines_normal_min, self.num_lines_normal_max + 1)
        self.sin_stop = 0.48
    
    @classmethod
    def sample_bb_dims(cls, rng, from_height):
        return super(Decoration, cls).sample_bb_dims(rng, from_height)
    
    @classmethod
    def update(cls, height_min, height_max, width_gain_min, width_gain_max, corner_radius_gain_min, corner_radius_gain_max,
              connector_height_gain_min, connector_height_gain_max, connector_width_gain_min, connector_width_gain_max, shaded_connector_prob,
              shaded_dock_fill_dist_gain_min, shaded_dock_fill_dist_gain_max, prob_lines, num_lines_parallel_min, num_lines_parallel_max,
              num_lines_normal_min, num_lines_normal_max):
        super(Inflatable, cls).update(height_min, height_max, width_gain_min, width_gain_max)
        cls.line_choice = pick_random_element(cls.line_types, prob_lines)
        cls.corner_radius_gain = np.random.uniform(corner_radius_gain_min, corner_radius_gain_max)
        cls.dock_height_gain_min = connector_height_gain_min
        cls.dock_height_gain_max = connector_height_gain_max
        cls.dock_width_gain_min = connector_width_gain_min
        cls.dock_width_gain_max = connector_width_gain_max
        cls.shaded_connector = np.random.rand() < shaded_connector_prob
        cls.shaded_dock_fill_dist_gain = np.random.uniform(shaded_dock_fill_dist_gain_min, shaded_dock_fill_dist_gain_max)
        cls.num_lines_parallel_min = num_lines_parallel_min
        cls.num_lines_parallel_max = num_lines_parallel_max
        cls.num_lines_normal_min = num_lines_normal_min
        cls.num_lines_normal_max = num_lines_normal_max
        
    def draw(self):
        sketch = self.init_sketch()
        
        if self.shaded_connector:
            sketch.sketch(draw_shaded_rect(0.5 * self.connector_width, 0, self.connector_width, self.connector_height, 
                                           fill_distance=self.shaded_connector_fill_dist))
        else:
            sketch.rect(0.5 * self.connector_width, 0, self.connector_width, self.connector_height, mode="center")
        sketch.translate(self.connector_width + 0.5 * self.capsule_width, 0)
        sketch.rect(0, 0, self.capsule_width, self.height, self.corner_radius, mode="center")
        
        if self.line_choice == "PARALLEL":
            ys = 0.5 * self.height * np.sin(np.pi * np.linspace(-self.sin_stop, self.sin_stop, num=self.num_lines_parallel))
            for y in ys:
                if np.abs(y) > 0.5 * self.height - self.corner_radius:
                    x = 0.5 * self.capsule_width - self.corner_radius * \
                        (1.0 - np.cos(np.arcsin((np.abs(y) - 0.5 * self.height + self.corner_radius) / self.corner_radius)))
                else:
                    x = 0.5 * self.capsule_width
                sketch.line(-x, y, x, y)
        elif self.line_choice == "NORMAL":
            xs = np.linspace(-0.5 * self.width, 0.5 * self.width, num=self.num_lines_normal)
            for x in xs:
                # sketch.arc()
                pass
                # TODO
            
        return sketch

    
class StationGenerator:
    def __init__(self, width, height, module_types, module_type_probs, probs_modules_parallel, probs_modules_normal, prob_connector_parallel_match_height,
                 weight_continue_same_dir=1.0):
        self.width = width
        self.height = height
        
        self.rng = default_rng()
        
        self.modules = []
        self.bounding_geometry = Point()  # empty geometry
        
        self.module_types = module_types
        self.n_main_module_types = len(self.module_types)
        self.module_type_to_idx = dict(zip(self.module_types.keys(), range(self.n_main_module_types)))
        self.module_type_probs = module_type_probs
        
        self.prob_connector_parallel_match_height = prob_connector_parallel_match_height
        self.probs_modules_parallel = probs_modules_parallel
        self.probs_modules_normal = probs_modules_normal
        self.weight_continue_same_dir = weight_continue_same_dir
    
    def get_bounding_box(self):
        return sh.box(-0.5 * self.width, -0.5 * self.height, 0.5 * self.width, 0.5 * self.height)
        
    def add_module(self, module, prev_module=None):
        self.modules.append(module)
        
        # Add bounding box of new module to overall bounding geometry:
        bb = module.get_bounding_box()
        self.bounding_geometry = sh.union(self.bounding_geometry, bb)
        
        # Remove no longer open points from previous module:
        # TODO: optional margin around the BB when removing points
        if prev_module is not None:
            line = module.get_edge_line()
            prev_module.open_points[module.direction] = sh.difference(prev_module.open_points[module.direction], line)
            if prev_module.open_points[module.direction].is_empty:  # remove side if all points are removed
                prev_module.open_points.pop(module.direction, None)
    
    def get_open_sides(self):
        sides, weights = [], []
        for idx, module in enumerate(self.modules):
            if module.open_points is not None:
                for dir in module.open_points.keys():
                    weight = self.weight_continue_same_dir if dir == module.direction else 1.0
                    weights.append(weight)
                    sides.append((idx, dir))
        return sides, weights
    
    def pick_random_submodule(self, from_submodule, dir):
        from_module_class = type(from_submodule).__base__
        
        from_module_idx = self.module_type_to_idx[from_module_class]
        if directions_are_normal(dir, from_submodule.direction):
            probs = self.probs_modules_normal[from_module_idx]
        else:
            probs = self.probs_modules_parallel[from_module_idx]
        
        module = pick_random_element(list(self.module_types.keys()), probs)
        submodule = pick_random_element(self.module_types[module], self.module_type_probs[module])
        return submodule
    
    def generate(self, num_tries, num_consec_fails_max=50):
        x, y, = 0.0, 0.0  # start in zero
        from_module = None
        consec_fails = 0
        for i in range(num_tries):
            if consec_fails >= num_consec_fails_max:  # number of consecutive fails termination criteria
                print("Termination: number of consecutive fails reached.")
                break
            
            if i > 0:
                # Sample random (but weighted) side:
                sides, side_weights = self.get_open_sides()
                if len(side_weights) == 0:  # exit if no options left
                    print("Termination: no side options left.")
                    break
                side_probs = normalize_vec_to_sum_one(side_weights)
                idx, dir = pick_random_element(sides, side_probs)
                
                from_module = self.modules[idx]
                point = random.choice(get_points_iterable(from_module.open_points[dir]))
                x, y = point.x, point.y
                
                module_class = self.pick_random_submodule(from_module, dir)
                
                # Pick random width and height:
                # TODO: if this gets more complicated build args dict and input **args instead
                if issubclass(module_class, Capsule) and isinstance(from_module, Connector):  # if capsule coming from connector
                    width, height = module_class.sample_bb_dims(self.rng, from_height=from_module.end_height, match_from_height=True)
                elif issubclass(module_class, Connector):  # if connector
                    are_normal = directions_are_normal(dir, from_module.direction)
                    if are_normal:
                        width, height = module_class.sample_bb_dims(self.rng, from_module.width, limit_height_by_from_height=True)
                    elif self.rng.random() < self.prob_connector_parallel_match_height:
                        width, height = module_class.sample_bb_dims(self.rng, from_module.height, match_from_height=True)
                    else:
                        width, height = module_class.sample_bb_dims(self.rng, from_module.height, limit_height_by_from_height=True)
                else:
                    width, height = module_class.sample_bb_dims(self.rng, from_module.height)
                module = module_class(x, y, width, height, dir, from_module=from_module)
            else:
                module_class = pick_random_element(self.module_types[Capsule], self.module_type_probs[Capsule])
                dir = random.choice(list(Direction))
                
                width, height = module_class.sample_bb_dims(self.rng, from_height=None)
                module = module_class(x, y, width, height, dir, from_module=None, allow_all_dirs=True)
        
            # Check if module fits in bounding geometry:
            intersects_bounding_geom = self.bounding_geometry.intersects(module.get_bounding_box(shrink=1e-4))
            
            # Check if module fits in outer bounding box:
            inside_outer_bb = self.get_bounding_box().contains(module.get_bounding_box())
            
            if inside_outer_bb and not intersects_bounding_geom:
                consec_fails = 0
                self.add_module(module, prev_module=from_module)
            else:
                consec_fails += 1

    def draw_outer_bounding_box(self, vsk):
        vsk.stroke(2)
        vsk.rect(0, 0, self.width, self.height, mode="center")
        vsk.stroke(1)
        
    def draw_bounding_boxes(self, vsk, arrow_length=3e-1, text_offset=0.2):
        vsk.stroke(2)
        for idx, module in enumerate(self.modules):
            vsk.sketch(module.draw_bounding_box())
            
            text_offset_curr = text_offset if module.direction == Direction.UP else -text_offset
            vsk.text(f"{idx}.{type(module).__name__[:2]}", module.x_center, 
                     module.y_center + text_offset_curr, size=0.15, align="center")
            
            with vsk.pushMatrix():  # draw arrow to show direction
                angle = direction_to_angle(module.direction)
                vsk.translate(module.x_center, module.y_center)
                vsk.rotate(-angle)
                vsk.line(0, 0, arrow_length, 0.0)
                vsk.line(arrow_length, 0.0, arrow_length - 0.3 * arrow_length / np.sqrt(2),
                         0.3 * arrow_length / np.sqrt(2))
                vsk.line(arrow_length, 0.0, arrow_length - 0.3 * arrow_length / np.sqrt(2),
                         -0.3 * arrow_length / np.sqrt(2))
        vsk.stroke(1)
            
    def draw_open_points(self, vsk):
        vsk.stroke(4)
        for module in self.modules:
            if module.open_points is not None:
                for points in module.open_points.values():
                    for point in get_points_iterable(points):
                        vsk.circle(point.x, point.y, radius=2e-2)
        vsk.stroke(1)
        
    def draw(self, vsk, draw_modules=True, draw_floating_connectors=True, debug=False):
        if draw_modules:
            for module in self.modules:
                if not draw_floating_connectors and isinstance(module, Connector) and len(module.open_points) > 0:
                    pass
                else:
                    vsk.sketch(module.draw())
        
        if debug:
            vsk.circle(0, 0, radius=1e-1)  # origin
            self.draw_outer_bounding_box(vsk)
            self.draw_bounding_boxes(vsk)
            self.draw_open_points(vsk)  


class Node:
    def __init__(self, drawing, type, x, y, width, height, layer, bottom=False):
        self.drawing = drawing
        self.type = type
        self.x, self.y = x, y
        self.width, self.height = width, height
        self.layer = layer
        self.bottom = bottom
        self._children = []
    
    def add_child(self, node):
        self._children.append(node)
    
    # def draw(self, vsk):
    #     if self.width > 0 and self.height > 0:
    #         if self.drawing.debug:
    #             vsk.stroke(2)
    #             vsk.rect(self.x, self.y, self.width, self.height, mode="center")
    #             vsk.stroke(1)
                
    #         vsk.rect(self.x, self.y, self.width, self.height, mode="center")
                            
    def __iter__(self):
        return iter(self._children)

    def breadth_first(self):
        q = [self]
        while q:
            n = q.pop(0)
            yield n
            for c in n._children:
                q.append(c)


class QuadTree:
    def __init__(self, drawing, width, height, outer_padding, inner_padding, panel_double_prob=0.6, 
                 panel_two_doubles_prob=0.6, panel_frame_prob=0.9, x=0.0, y=0.0, layers=2):
        self.drawing = drawing
        self.outer_padding = outer_padding
        self.inner_padding = inner_padding
        self.width = width
        self.height = height
        self.panel_double_prob = panel_double_prob
        self.panel_two_doubles_prob = panel_two_doubles_prob
        self.panel_frame_prob = panel_frame_prob
        self.x = x
        self.y = y
        self.layers = layers
        
        self.root = self.generate_tree(self.layers)
    
    def generate_children(self, node, layers):
        picks = []
        
        pick_double = np.random.random_sample() < self.panel_double_prob
        if pick_double:
            double_sections = [[-1,0], [1,0], [0,-1], [0,1]]
            pick = double_sections[np.random.randint(len(double_sections))]
            x, y = pick
            picks.append(pick)
            pick_two_doubles = pick_double and np.random.random_sample() < self.panel_two_doubles_prob
            if pick_two_doubles:
                picks.append([-x, -y])
            else:
                if x != 0:
                    picks.append([-x, 1])
                    picks.append([-x, -1])
                else:
                    picks.append([1, -y])
                    picks.append([-1, -y]) 
        else:
            picks += [[-1,-1], [-1,1], [1,-1], [1,1]]
            
        for pick in picks:
            is_double = (pick[0] == 0) or (pick[1] == 0)
            x = 0.25 * node.width * pick[0]
            y = 0.25 * node.height * pick[1]
            
            if pick[0] == 0:
                child_width = node.width - 2 * self.inner_padding
            else:
                child_width = 0.5 * node.width - 2 * self.inner_padding
            if pick[1] == 0:
                child_height = node.height - 2 * self.inner_padding
            else:
                child_height = 0.5 * node.height - 2 * self.inner_padding
            
            if child_height > 1e-2 and child_width > 1e-2:
                stop_prob = 0.5
                do_stop = (np.random.random_sample() < stop_prob) or (node.layer == layers - 1)
                bottom = do_stop or is_double
                
                child = Node(self.drawing, "...", x=node.x+x, y=node.y+y, width=child_width,
                            height=child_height, layer=node.layer+1, bottom=bottom)
                if not bottom:
                    self.generate_children(child, layers)
                node.add_child(child)
    
    def generate_tree(self, layers):
        inner_width = self.width - self.outer_padding
        inner_height = self.height - self.outer_padding
        
        root = Node(self.drawing, "root", x=0, y=0, width=inner_width, height=inner_height, layer=0)
        
        self.generate_children(root, layers)
        
        return root
    
    def draw(self, vsk):
        with vsk.pushMatrix():
            vsk.translate(self.x, self.y)
            for node in self.root.breadth_first():
                node.draw(vsk)
                    

class SpacestationSketch(vsketch.SketchClass):
    WIDTH_FULL = 21
    HEIGHT_FULL = 29.7
    
    module_types = {Capsule: [CapsuleVariation1, CapsuleMultiWindow, Capsule3D, CapsuleParallelLines, 
                                CapsuleNormalLines, SquareCapsule],
                        Connector: [ConnectorTrapezoid, ConnectorTrapezoidExtended, ConnectorSimple],
                        SolarPanel: [SolarPanelSingle, SolarPanelDouble],
                        Decoration: [Antenna, DockingBaySimple, DockingBay, Boxes, Inflatable]}
    
    draw_modules = vsketch.Param(True)
    debug = vsketch.Param(True)
    occult = vsketch.Param(False)
    draw_floating_connectors = vsketch.Param(False)
    scale = vsketch.Param(0.6)
    
    num_tries = vsketch.Param(200, min_value=1)
    num_consec_fails_max = vsketch.Param(50, min_value=1)
    
    grid_type = vsketch.Param("GRID", choices=["GRID", "QUAD"])
    n_x = vsketch.Param(2, min_value=1)
    n_y = vsketch.Param(3, min_value=1)
    
    outer_pad = vsketch.Param(0.5)
    grid_pad = vsketch.Param(0.3)
    quad_layers = vsketch.Param(2)
    
    weight_continue_same_dir = vsketch.Param(10.0, min_value=0.0)
    
    prob_connector_parallel_match_height = vsketch.Param(0.9, min_value=0.0, max_value=1.0)
    
    # Module transition probs:
    prob_capsule_capsule_parallel = vsketch.Param(1.0, min_value=0)
    prob_capsule_connector_parallel = vsketch.Param(2.0, min_value=0)
    prob_capsule_solar_parallel = vsketch.Param(0.4, min_value=0)
    prob_capsule_decoration_parallel = vsketch.Param(1.0, min_value=0)

    prob_capsule_capsule_normal = vsketch.Param(1.0, min_value=0)
    prob_capsule_connector_normal = vsketch.Param(3.0, min_value=0)
    prob_capsule_solar_normal = vsketch.Param(3.0, min_value=0)
    prob_capsule_decoration_normal = vsketch.Param(1.0, min_value=0)
    
    # Number of choices to pick of each module type:
    num_capsule_choices = vsketch.Param(3, min_value=1)
    num_connector_choices = vsketch.Param(2, min_value=1)
    num_solar_choices = vsketch.Param(1, min_value=1)
    num_decorations_choices = vsketch.Param(3, min_value=1)
    
    # Probability of picking module subtype in list of modules:
    prob_choice_capsule_variation_1 = vsketch.Param(1.0, min_value=0.0)
    prob_choice_capsule_multi_window = vsketch.Param(1.0, min_value=0.0)
    prob_choice_capsule_3d = vsketch.Param(1.0, min_value=0.0)
    prob_choice_capsule_parallel_lines = vsketch.Param(1.0, min_value=0.0)
    prob_choice_capsule_normal_lines = vsketch.Param(1.0, min_value=0.0)
    prob_choice_capsule_square = vsketch.Param(3.0, min_value=0.0)
    
    prob_choice_connector_trapezoid = vsketch.Param(1.0, min_value=0.0)
    prob_choice_connector_trap_ext = vsketch.Param(1.0, min_value=0.0)
    prob_choice_connector_simple = vsketch.Param(1.0, min_value=0.0)
    
    prob_choice_solar_single = vsketch.Param(1.0, min_value=0.0)
    prob_choice_solar_double = vsketch.Param(2.0, min_value=0.0)
    
    prob_choice_decoration_antenna = vsketch.Param(3.0, min_value=0.0)
    prob_choice_decoration_dock_simple = vsketch.Param(1.0, min_value=0.0)
    prob_choice_decoration_dock = vsketch.Param(1.0, min_value=0.0)
    prob_choice_decoration_boxes = vsketch.Param(3.0, min_value=0.0)
    prob_choice_decoration_inflatable = vsketch.Param(1.0, min_value=0.0)
    
    # Probability of picking module subtypes within drawing:
    prob_capsule_variation_1 = vsketch.Param(1.0, min_value=0.0)
    prob_capsule_multi_window = vsketch.Param(0.5, min_value=0.0)
    prob_capsule_3d = vsketch.Param(0.5, min_value=0.0)
    prob_capsule_parallel_lines = vsketch.Param(0.5, min_value=0.0)
    prob_capsule_normal_lines = vsketch.Param(0.5, min_value=0.0)
    prob_capsule_square = vsketch.Param(0.2, min_value=0.0)
    
    prob_connector_trapezoid = vsketch.Param(1.0, min_value=0.0)
    prob_connector_trap_ext = vsketch.Param(1.0, min_value=0.0)
    prob_connector_simple = vsketch.Param(1.0, min_value=0.0)
    
    prob_solar_single = vsketch.Param(1.0, min_value=0.0)
    prob_solar_double = vsketch.Param(1.0, min_value=0.0)
    
    prob_decoration_antenna = vsketch.Param(0.5, min_value=0.0)
    prob_decoration_dock_simple = vsketch.Param(1.0, min_value=0.0)
    prob_decoration_dock = vsketch.Param(2.0, min_value=0.0)
    prob_decoration_boxes = vsketch.Param(1.0, min_value=0.0)
    prob_decoration_inflatable = vsketch.Param(1.0, min_value=0.0)
    
    # Module params:
    capsule_height_min = vsketch.Param(1.0, min_value=0)
    capsule_height_max = vsketch.Param(2.0, min_value=0)
    capsule_width_gain_min = vsketch.Param(1.0, min_value=0)
    capsule_width_gain_max = vsketch.Param(3.0, min_value=0)
    
    capsule_multi_window_circle_prob = vsketch.Param(1.0, min_value=0)
    capsule_multi_window_square_prob = vsketch.Param(1.0, min_value=0)
    capsule_multi_window_square_rounded_prob = vsketch.Param(1.5, min_value=0)
    capsule_multi_window_window_size_gain_min = vsketch.Param(0.18, min_value=0)
    capsule_multi_window_window_size_gain_max = vsketch.Param(0.4, min_value=0)
    capsule_multi_window_windows_size_gain_min = vsketch.Param(0.2, min_value=0)
    capsule_multi_window_windows_size_gain_max = vsketch.Param(0.6, min_value=0)
    capsule_multi_window_window_dist_gain_min = vsketch.Param(0.05, min_value=0)
    capsule_multi_window_window_dist_gain_max = vsketch.Param(0.2, min_value=0)
    capsule_multi_window_rounded_radius_gain_min = vsketch.Param(0.05, min_value=0)
    capsule_multi_window_rounded_radius_gain_max = vsketch.Param(0.3, min_value=0)
    capsule_multi_window_no_lines_prob = vsketch.Param(2.0, min_value=0)
    capsule_multi_window_parallel_lines_prob = vsketch.Param(1.0, min_value=0)
    capsule_multi_window_parallel_line_dist_gain_min = vsketch.Param(0.1, min_value=0)
    capsule_multi_window_parallel_line_dist_gain_max = vsketch.Param(0.2, min_value=0)
    capsule_multi_window_normal_lines_prob = vsketch.Param(0.75, min_value=0)
    capsule_multi_window_box_prob = vsketch.Param(0.75, min_value=0)
    
    capsule_normal_lines_prob_random = vsketch.Param(0.3, min_value=0)
    capsule_normal_lines_prob_double_thin = vsketch.Param(0.15, min_value=0)
    capsule_normal_lines_prob_double_flat = vsketch.Param(0.1, min_value=0)
    capsule_normal_lines_prob_double_multi = vsketch.Param(0.1, min_value=0)
    capsule_normal_lines_prob_double_multi_random = vsketch.Param(0.1, min_value=0)
    capsule_normal_lines_prob_double_shaded = vsketch.Param(0.2, min_value=0)
    capsule_normal_lines_prob_double_black = vsketch.Param(0.15, min_value=0)
    capsule_normal_lines_num_lines_rand_min = vsketch.Param(1, min_value=0)
    capsule_normal_lines_num_lines_rand_max = vsketch.Param(9, min_value=0)
    capsule_normal_lines_double_offset_gain_min = vsketch.Param(0.1, min_value=0)
    capsule_normal_lines_double_offset_gain_max = vsketch.Param(0.3, min_value=0)
    capsule_normal_lines_double_dist_gain_min = vsketch.Param(0.02, min_value=0)
    capsule_normal_lines_double_dist_gain_max = vsketch.Param(0.2, min_value=0)
    capsule_normal_lines_num_lines_multi_min = vsketch.Param(3, min_value=0)
    capsule_normal_lines_num_lines_multi_max = vsketch.Param(6, min_value=0)
    capsule_normal_lines_double_multi_dist_gain_min = vsketch.Param(0.1, min_value=0)
    capsule_normal_lines_double_multi_dist_gain_max = vsketch.Param(0.3, min_value=0)
    capsule_normal_lines_double_shaded_dist_min = vsketch.Param(0.02, min_value=0)
    capsule_normal_lines_double_shaded_dist_max = vsketch.Param(0.2, min_value=0)
    
    capsule_square_height_min = vsketch.Param(0.6, min_value=0)
    capsule_square_height_max = vsketch.Param(1.6, min_value=0)
    capsule_square_border_prob = vsketch.Param(0.6, min_value=0)
    capsule_square_border_gain_min = vsketch.Param(0.95, min_value=0)
    capsule_square_border_gain_max = vsketch.Param(0.8, min_value=0)
    capsule_square_rounded_corners_prob = vsketch.Param(0.3, min_value=0)
    capsule_square_cross_prob = vsketch.Param(0.5, min_value=0)
    capsule_square_shaded_circle_prob = vsketch.Param(0.5, min_value=0)
    capsule_square_corner_radius_gain_min = vsketch.Param(0.05, min_value=0)
    capsule_square_corner_radius_gain_max = vsketch.Param(0.2, min_value=0)
    capsule_square_outer_circle_gain_min = vsketch.Param(0.5, min_value=0)
    capsule_square_outer_circle_gain_max = vsketch.Param(0.7, min_value=0)
    capsule_square_inner_circle_gain_min = vsketch.Param(0.6, min_value=0)
    capsule_square_inner_circle_gain_max = vsketch.Param(0.95, min_value=0)
    capsule_square_num_lines_shaded_circle_min = vsketch.Param(6, min_value=0)
    capsule_square_num_lines_shaded_circle_max = vsketch.Param(20, min_value=0)
    
    capsule_3d_num_lines_min = vsketch.Param(5, min_value=0)
    capsule_3d_num_lines_max = vsketch.Param(20, min_value=0)
    
    capsule_parallel_lines_num_lines_min = vsketch.Param(1, min_value=0)
    capsule_parallel_lines_num_lines_max = vsketch.Param(5, min_value=0)
    
    
    connector_height_min = vsketch.Param(1.0, min_value=0)
    connector_height_max = vsketch.Param(2.0, min_value=0)
    connector_from_height_gain_min = vsketch.Param(0.833, min_value=0)
    connector_from_height_gain_max = vsketch.Param(1.2, min_value=0)
    connector_width_gain_min = vsketch.Param(0.2, min_value=0)
    connector_width_gain_max = vsketch.Param(0.4, min_value=0)
    
    connector_trap_ext_union_prob = vsketch.Param(1.0, min_value=0)
    connector_trap_ext_empty_prob = vsketch.Param(1.0, min_value=0)
    connector_trap_ext_shaded_prob = vsketch.Param(1.0, min_value=0)
    connector_trap_ext_height_min = vsketch.Param(1.0, min_value=0)
    connector_trap_ext_height_max = vsketch.Param(2.0, min_value=0)
    connector_trap_ext_from_height_gain_min = vsketch.Param(0.833, min_value=0)
    connector_trap_ext_from_height_gain_max = vsketch.Param(1.2, min_value=0)
    connector_trap_ext_width_gain_min = vsketch.Param(0.2, min_value=0)
    connector_trap_ext_width_gain_max = vsketch.Param(0.3, min_value=0)
    connector_trap_ext_height_gain_min = vsketch.Param(0.6, min_value=0)
    connector_trap_ext_height_gain_max = vsketch.Param(0.8, min_value=0)
    connector_trap_ext_flat_width_gain_min = vsketch.Param(0.4, min_value=0)
    connector_trap_ext_flat_width_gain_max = vsketch.Param(0.6, min_value=0)
    connector_trap_ext_shaded_dist_min = vsketch.Param(0.02, min_value=0)
    connector_trap_ext_shaded_dist_max = vsketch.Param(0.1, min_value=0)
    
    connector_simple_empty_prob = vsketch.Param(1.0, min_value=0)
    connector_simple_shaded_prob = vsketch.Param(1.0, min_value=0)
    connector_simple_filled_prob = vsketch.Param(1.0, min_value=0)
    connector_simple_height_min = vsketch.Param(1.0, min_value=0)
    connector_simple_height_max = vsketch.Param(2.0, min_value=0)
    connector_simple_from_height_gain_min = vsketch.Param(0.833, min_value=0)
    connector_simple_from_height_gain_max = vsketch.Param(1.2, min_value=0)
    connector_simple_width_gain_min = vsketch.Param(0.1, min_value=0)
    connector_simple_width_gain_max = vsketch.Param(0.15, min_value=0)
    connector_simple_height_gain_min = vsketch.Param(0.6, min_value=0)
    connector_simple_height_gain_max = vsketch.Param(0.8, min_value=0)
    connector_simple_shaded_dist_min = vsketch.Param(0.06, min_value=0)
    connector_simple_shaded_dist_max = vsketch.Param(0.12, min_value=0)
    
    solar_height_min = vsketch.Param(1.4, min_value=0)
    solar_height_max = vsketch.Param(1.9, min_value=0)
    solar_width_gain_min = vsketch.Param(4.0, min_value=0)
    solar_width_gain_max = vsketch.Param(7.0, min_value=0)
    solar_panel_dist_x_min = vsketch.Param(0.10, min_value=0)
    solar_panel_dist_x_max = vsketch.Param(0.20, min_value=0)
    solar_panel_num_y_min = vsketch.Param(2, min_value=0)
    solar_panel_num_y_max = vsketch.Param(4, min_value=0)

    solar_panel_double_connector_width_min = vsketch.Param(0.12, min_value=0)
    solar_panel_double_connector_width_max = vsketch.Param(0.3, min_value=0)  
    solar_panel_double_connector_height_min = vsketch.Param(0.05, min_value=0)
    solar_panel_double_connector_height_max = vsketch.Param(0.2, min_value=0)  
    solar_panel_double_panel_dist_min = vsketch.Param(0.1, min_value=0)
    solar_panel_double_panel_dist_max = vsketch.Param(0.3, min_value=0)    
    solar_panel_double_inset_min = vsketch.Param(0.0, min_value=0)
    solar_panel_double_inset_max = vsketch.Param(0.1, min_value=0)
    solar_panel_double_multi_beam_prob = vsketch.Param(0.3, min_value=0)
    solar_panel_double_n_beams_extra_min = vsketch.Param(1, min_value=0)
    solar_panel_double_n_beams_extra_max = vsketch.Param(5, min_value=0)

    antenna_height_min = vsketch.Param(0.1, min_value=0)
    antenna_height_max = vsketch.Param(0.15, min_value=0)
    antenna_width_gain_min = vsketch.Param(1.0, min_value=0)
    antenna_width_gain_max = vsketch.Param(3.0, min_value=0)
    antenna_empty_prob = vsketch.Param(0.5, min_value=0)
    antenna_dot_prob = vsketch.Param(0.5, min_value=0)
    antenna_square_prob = vsketch.Param(0.5, min_value=0)
    antenna_dot_radius_min = vsketch.Param(0.01, min_value=0)
    antenna_dot_radius_max = vsketch.Param(0.03, min_value=0)
    
    dock_simple_height_min = vsketch.Param(0.3, min_value=0)
    dock_simple_height_max = vsketch.Param(1.0, min_value=0)
    dock_simple_width_gain_min = vsketch.Param(0.07, min_value=0)
    dock_simple_width_gain_max = vsketch.Param(0.15, min_value=0)
    dock_simple_black_prob = vsketch.Param(0.5, min_value=0)
    dock_simple_shaded_dist_min = vsketch.Param(0.02, min_value=0)
    dock_simple_shaded_dist_max = vsketch.Param(0.1, min_value=0)
        
    dock_height_min = vsketch.Param(1.0, min_value=0)
    dock_height_max = vsketch.Param(2.0, min_value=0)
    dock_width_gain_min = vsketch.Param(0.4, min_value=0)
    dock_width_gain_max = vsketch.Param(0.7, min_value=0)
    dock_union_middle_prob = vsketch.Param(0.5, min_value=0) 
    dock_flat_end_height_gain_min = vsketch.Param(0.45, min_value=0)
    dock_flat_end_height_gain_max = vsketch.Param(0.6, min_value=0)
    dock_end_height_gain_min = vsketch.Param(0.7, min_value=0)
    dock_end_height_gain_max = vsketch.Param(0.9, min_value=0)
    dock_start_frac_min = vsketch.Param(0.03, min_value=0)
    dock_start_frac_max = vsketch.Param(0.15, min_value=0)
    dock_end_frac_min = vsketch.Param(0.05, min_value=0)
    dock_end_frac_max = vsketch.Param(0.12, min_value=0)
    dock_flat_start_frac_min = vsketch.Param(0.2, min_value=0)
    dock_flat_start_frac_max = vsketch.Param(0.3, min_value=0)
    dock_flat_end_frac_min = vsketch.Param(0.1, min_value=0)
    dock_flat_end_frac_max = vsketch.Param(0.2, min_value=0)
    
    boxes_height_min = vsketch.Param(1.0, min_value=0)
    boxes_height_max = vsketch.Param(2.0, min_value=0)
    boxes_width_gain_min = vsketch.Param(0.15, min_value=0)
    boxes_width_gain_max = vsketch.Param(0.25, min_value=0)
    boxes_prob_multi = vsketch.Param(0.5, min_value=0)
    boxes_prob_line_on_box = vsketch.Param(0.4, min_value=0)
    boxes_prob_box_on_box = vsketch.Param(0.4, min_value=0)
    boxes_num_box_multi_min = vsketch.Param(1, min_value=0)
    boxes_num_box_multi_max = vsketch.Param(1, min_value=0)
    boxes_box_width_gain_min = vsketch.Param(0.7, min_value=0)
    boxes_box_width_gain_max = vsketch.Param(1.0, min_value=0)
    boxes_box_height_gain_min = vsketch.Param(0.4, min_value=0)
    boxes_box_height_gain_max = vsketch.Param(0.8, min_value=0)
    boxes_box_box_width_gain_min = vsketch.Param(0.1, min_value=0) 
    boxes_box_box_width_gain_max = vsketch.Param(0.5, min_value=0) 
    boxes_box_box_height_gain_min = vsketch.Param(0.1, min_value=0) 
    boxes_box_box_height_gain_max = vsketch.Param(0.5, min_value=0) 
    boxes_line_length_gain_min = vsketch.Param(0.1, min_value=0) 
    boxes_line_length_gain_max = vsketch.Param(0.4, min_value=0) 
    
    inflatable_height_min = vsketch.Param(1.75, min_value=0)
    inflatable_height_max = vsketch.Param(2.5, min_value=0)
    inflatable_width_gain_min = vsketch.Param(1.4, min_value=0)
    inflatable_width_gain_max = vsketch.Param(1.8, min_value=0)
    inflatable_corner_radius_gain_min = vsketch.Param(0.35, min_value=0)
    inflatable_corner_radius_gain_max = vsketch.Param(0.45, min_value=0)
    inflatable_dock_height_gain_min = vsketch.Param(0.25, min_value=0)
    inflatable_dock_height_gain_max = vsketch.Param(0.45, min_value=0)
    inflatable_dock_width_gain_min = vsketch.Param(0.2, min_value=0)
    inflatable_dock_width_gain_max = vsketch.Param(0.4, min_value=0)
    inflatable_shaded_dock_prob = vsketch.Param(0.5, min_value=0)
    inflatable_shaded_dock_fill_dist_gain_min = vsketch.Param(0.05, min_value=0)
    inflatable_shaded_dock_fill_dist_gain_max = vsketch.Param(0.2, min_value=0)
    inflatable_empty_prob = vsketch.Param(0.5, min_value=0)
    inflatable_parallel_lines_prob = vsketch.Param(0.5, min_value=0)
    inflatable_normal_lines_prob = vsketch.Param(0.5, min_value=0)
    inflatable_num_lines_parallel_min = vsketch.Param(7, min_value=0)
    inflatable_num_lines_parallel_max = vsketch.Param(12, min_value=0)
    inflatable_num_lines_normal_min = vsketch.Param(3, min_value=0)
    inflatable_num_lines_normal_max = vsketch.Param(8, min_value=0)
    
    def init_drawing(self, vsk):
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale, self.scale)
        print("\nRunning...")
        
    def init_probs(self):
        probs_parallel = np.array([[self.prob_capsule_capsule_parallel, self.prob_capsule_connector_parallel,
                                    self.prob_capsule_solar_parallel, self.prob_capsule_decoration_parallel],
                                   [1.0, 0.0, 0.0, 0.0],
                                   4 * [np.nan],
                                   4 * [np.nan]])
        self.probs_modules_parallel = normalize_mat_to_row_sum_one(probs_parallel)
        
        probs_normal = np.array([[self.prob_capsule_capsule_normal, self.prob_capsule_connector_normal,
                                  self.prob_capsule_solar_normal, self.prob_capsule_decoration_normal],
                                 4 * [np.nan],
                                 4 * [np.nan],
                                 4 * [np.nan]])
        self.probs_modules_normal = normalize_mat_to_row_sum_one(probs_normal)
        
        self.num_module_types_choices = {Capsule: self.num_capsule_choices, 
                                         Connector: self.num_connector_choices,
                                         SolarPanel: self.num_solar_choices,
                                         Decoration: self.num_decorations_choices}
        
        self.module_type_probs_choice = {Capsule: normalize_vec_to_sum_one([self.prob_choice_capsule_variation_1, self.prob_choice_capsule_multi_window, 
                                                                            self.prob_choice_capsule_3d, self.prob_choice_capsule_parallel_lines,
                                                                            self.prob_choice_capsule_normal_lines, self.prob_choice_capsule_square]),
                                  Connector: normalize_vec_to_sum_one([self.prob_choice_connector_trapezoid, self.prob_choice_connector_trap_ext,
                                                                       self.prob_choice_connector_simple]),
                                  SolarPanel: normalize_vec_to_sum_one([self.prob_choice_solar_single, self.prob_choice_solar_double]),
                                  Decoration: normalize_vec_to_sum_one([self.prob_choice_decoration_antenna, self.prob_choice_decoration_dock_simple,
                                                                        self.prob_choice_decoration_dock, self.prob_choice_decoration_boxes,
                                                                        self.prob_choice_decoration_inflatable])}
         
        self.module_type_probs = {Capsule: normalize_vec_to_sum_one([self.prob_capsule_variation_1, self.prob_capsule_multi_window, self.prob_capsule_3d, 
                                                                     self.prob_capsule_parallel_lines, self.prob_capsule_normal_lines, 
                                                                     self.prob_capsule_square]),
                                  Connector: normalize_vec_to_sum_one([self.prob_connector_trapezoid, self.prob_connector_trap_ext,
                                                                       self.prob_connector_simple]),
                                  SolarPanel: normalize_vec_to_sum_one([self.prob_solar_single, self.prob_solar_double]),
                                  Decoration: normalize_vec_to_sum_one([self.prob_decoration_antenna, self.prob_decoration_dock_simple,
                                                                        self.prob_decoration_dock, self.prob_decoration_boxes,
                                                                        self.prob_decoration_inflatable])}
        
        probs_capsule_normal_line = np.array([self.capsule_normal_lines_prob_random, self.capsule_normal_lines_prob_double_thin, 
                                              self.capsule_normal_lines_prob_double_flat, self.capsule_normal_lines_prob_double_multi,
                                              self.capsule_normal_lines_prob_double_multi_random,
                                              self.capsule_normal_lines_prob_double_shaded, self.capsule_normal_lines_prob_double_black])
        self.capsule_normal_lines_probs = normalize_vec_to_sum_one(probs_capsule_normal_line)
        
        connector_trap_ext_probs = np.array([self.connector_trap_ext_union_prob, self.connector_trap_ext_empty_prob,
                                           self.connector_trap_ext_shaded_prob])
        self.connector_trap_ext_probs = normalize_vec_to_sum_one(connector_trap_ext_probs)
        
        connector_simple_probs = np.array([self.connector_simple_empty_prob, self.connector_simple_shaded_prob,
                                           self.connector_simple_filled_prob])
        self.connector_simple_probs = normalize_vec_to_sum_one(connector_simple_probs)
    
        probs_antenna = np.array([self.antenna_empty_prob, self.antenna_dot_prob, self.antenna_square_prob])
        self.antenna_probs = normalize_vec_to_sum_one(probs_antenna)
        
        multi_window_window_probs = np.array([self.capsule_multi_window_circle_prob, self.capsule_multi_window_square_prob, 
                                                    self.capsule_multi_window_square_rounded_prob])
        self.capsule_multi_window_windows_probs = normalize_vec_to_sum_one(multi_window_window_probs)
    
        probs_multi_window_line_probs = np.array([self.capsule_multi_window_no_lines_prob, self.capsule_multi_window_parallel_lines_prob, 
                                                  self.capsule_multi_window_normal_lines_prob, self.capsule_multi_window_box_prob])
        self.capsule_multi_window_line_probs = normalize_vec_to_sum_one(probs_multi_window_line_probs)
        
        probs_inflatable_line_probs = np.array([self.inflatable_empty_prob, self.inflatable_parallel_lines_prob, 
                                                self.inflatable_parallel_lines_prob])
        self.inflatable_line_probs = normalize_vec_to_sum_one(probs_inflatable_line_probs)
    
    def resample_module_choices(self):
        # Pick subset of module subtypes for each module type: 
        self.module_types_curr = {}
        self.module_type_probs_curr = {}
        for module_class in (Capsule, Connector, SolarPanel, Decoration):
            idx_choices = np.random.choice(np.arange(len(self.module_types[module_class])), 
                                           size=self.num_module_types_choices[module_class],
                                           replace=False, p=self.module_type_probs_choice[module_class])
            self.module_types_curr[module_class] = [self.module_types[module_class][i] for i in idx_choices]
            self.module_type_probs_curr[module_class] = normalize_vec_to_sum_one(self.module_type_probs[module_class][idx_choices])
            
    def update_module_params(self):
        # Capsules:     
        Capsule.update(self.capsule_height_min, self.capsule_height_max, self.capsule_width_gain_min,
                       self.capsule_width_gain_max)
        CapsuleMultiWindow.update(self.capsule_multi_window_windows_probs, self.capsule_multi_window_window_size_gain_min, self.capsule_multi_window_window_size_gain_max, 
                                  self.capsule_multi_window_windows_size_gain_min, self.capsule_multi_window_windows_size_gain_max, self.capsule_multi_window_window_dist_gain_min, 
                                  self.capsule_multi_window_window_dist_gain_max, self.capsule_multi_window_rounded_radius_gain_min,
                                  self.capsule_multi_window_rounded_radius_gain_max, self.capsule_multi_window_line_probs, self.capsule_multi_window_parallel_line_dist_gain_min, 
                                  self.capsule_multi_window_parallel_line_dist_gain_max)
        Capsule3D.update(self.capsule_3d_num_lines_min, self.capsule_3d_num_lines_max)
        CapsuleParallelLines.update(self.capsule_parallel_lines_num_lines_min, self.capsule_parallel_lines_num_lines_max)
        CapsuleNormalLines.update(self.capsule_normal_lines_probs, self.capsule_normal_lines_num_lines_rand_min, 
                                  self.capsule_normal_lines_num_lines_rand_max, self.capsule_normal_lines_double_offset_gain_min, 
                                  self.capsule_normal_lines_double_offset_gain_max, self.capsule_normal_lines_double_dist_gain_min, 
                                  self.capsule_normal_lines_double_dist_gain_max, self.capsule_normal_lines_num_lines_multi_min, 
                                  self.capsule_normal_lines_num_lines_multi_max, self.capsule_normal_lines_double_multi_dist_gain_min, 
                                  self.capsule_normal_lines_double_multi_dist_gain_max, self.capsule_normal_lines_double_shaded_dist_min,
                                  self.capsule_normal_lines_double_shaded_dist_max)
        SquareCapsule.update(self.capsule_square_height_min, self.capsule_square_height_max, self.capsule_square_border_prob, 
                             self.capsule_square_cross_prob, self.capsule_square_shaded_circle_prob, self.capsule_square_rounded_corners_prob, 
                             self.capsule_square_corner_radius_gain_min, self.capsule_square_corner_radius_gain_max, self.capsule_square_border_gain_min, 
                             self.capsule_square_border_gain_max, self.capsule_square_outer_circle_gain_min, self.capsule_square_outer_circle_gain_max, 
                             self.capsule_square_inner_circle_gain_min, self.capsule_square_inner_circle_gain_max, self.capsule_square_num_lines_shaded_circle_min,
                             self.capsule_square_num_lines_shaded_circle_max)
        
        # Connectors:
        Connector.update(self.connector_height_min, self.connector_height_max, self.connector_from_height_gain_min,
                         self.connector_from_height_gain_max, self.connector_width_gain_min, self.connector_width_gain_max)
        ConnectorTrapezoidExtended.update(self.connector_trap_ext_height_min, self.connector_trap_ext_height_max,
                                          self.connector_trap_ext_from_height_gain_min, self.connector_trap_ext_from_height_gain_max,
                                          self.connector_trap_ext_width_gain_min, self.connector_trap_ext_width_gain_max,
                                          self.connector_trap_ext_height_gain_min, self.connector_trap_ext_height_gain_max, 
                                          self.connector_trap_ext_flat_width_gain_min, self.connector_trap_ext_flat_width_gain_max,
                                          self.connector_trap_ext_shaded_dist_min, self.connector_trap_ext_shaded_dist_max,
                                          self.connector_trap_ext_probs)
        ConnectorSimple.update(self.connector_simple_height_min, self.connector_simple_height_max, 
                               self.connector_simple_from_height_gain_min, self.connector_simple_from_height_gain_max, 
                               self.connector_simple_width_gain_min, self.connector_simple_width_gain_max,
                               self.connector_simple_height_gain_min, self.connector_simple_height_gain_max, 
                               self.connector_simple_shaded_dist_min, self.connector_simple_shaded_dist_max, 
                               self.connector_simple_probs)
        
        # Solar panels:
        SolarPanel.update(self.solar_height_min, self.solar_height_max, self.solar_width_gain_min,
                          self.solar_width_gain_max, self.solar_panel_num_y_min, self.solar_panel_num_y_max, 
                          self.solar_panel_dist_x_min, self.solar_panel_dist_x_max)
        SolarPanelDouble.update(self.solar_panel_double_connector_width_min, self.solar_panel_double_connector_width_max, 
                                self.solar_panel_double_connector_height_min, self.solar_panel_double_connector_height_max, 
                                self.solar_panel_double_panel_dist_min, self.solar_panel_double_panel_dist_max, 
                                self.solar_panel_double_inset_min, self.solar_panel_double_inset_max, self.solar_panel_double_multi_beam_prob,
                                self.solar_panel_double_n_beams_extra_min, self.solar_panel_double_n_beams_extra_max)

        # Decorations:
        Antenna.update(self.antenna_height_min, self.antenna_height_max, self.antenna_width_gain_min,
                       self.antenna_width_gain_max, self.antenna_probs, self.antenna_dot_radius_min,
                       self.antenna_dot_radius_max)        
        DockingBaySimple.update(self.dock_simple_height_min, self.dock_simple_height_max, self.dock_simple_width_gain_min,
                               self.dock_simple_width_gain_max, self.dock_simple_black_prob, self.dock_simple_shaded_dist_min,
                               self.dock_simple_shaded_dist_max)
        DockingBay.update(self.dock_height_min, self.dock_height_max, self.dock_width_gain_min,
                          self.dock_width_gain_max, self.dock_union_middle_prob, self.dock_flat_end_height_gain_min, 
                          self.dock_flat_end_height_gain_max, 
                          self.dock_end_height_gain_min, self.dock_end_height_gain_max, self.dock_start_frac_min, 
                          self.dock_start_frac_max, self.dock_end_frac_min, self.dock_end_frac_max,
                          self.dock_flat_start_frac_min, self.dock_flat_start_frac_max, self.dock_flat_end_frac_min,
                          self.dock_flat_end_frac_max)
        Boxes.update(self.boxes_height_min, self.boxes_height_max, self.boxes_width_gain_min, self.boxes_width_gain_max,
                     self.boxes_prob_multi, self.boxes_prob_box_on_box, self.boxes_prob_line_on_box, 
                     self.boxes_num_box_multi_min, self.boxes_num_box_multi_max,
                     self.boxes_box_width_gain_min, self.boxes_box_width_gain_max, self.boxes_box_height_gain_min,
                     self.boxes_box_height_gain_max, self.boxes_box_box_width_gain_min, self.boxes_box_box_width_gain_max, 
                     self.boxes_box_box_height_gain_min, self.boxes_box_box_height_gain_max, self.boxes_line_length_gain_min, 
                     self.boxes_line_length_gain_max)
        Inflatable.update(self.inflatable_height_min, self.inflatable_height_max, self.inflatable_width_gain_min, 
                          self.inflatable_width_gain_max, self.inflatable_corner_radius_gain_min, self.inflatable_corner_radius_gain_max,
                          self.inflatable_dock_height_gain_min, self.inflatable_dock_height_gain_max, self.inflatable_dock_width_gain_min, 
                          self.inflatable_dock_width_gain_max, self.inflatable_shaded_dock_prob, self.inflatable_shaded_dock_fill_dist_gain_min,
                          self.inflatable_shaded_dock_fill_dist_gain_max, self.inflatable_line_probs, self.inflatable_num_lines_parallel_min,
                          self.inflatable_num_lines_parallel_max, self.inflatable_num_lines_normal_min, self.inflatable_num_lines_normal_max)
        
    def draw(self, vsk: vsketch.Vsketch) -> None:
        self.init_drawing(vsk)
        self.init_probs()
        self.resample_module_choices()
        self.update_module_params()
        
        width = (self.WIDTH_FULL - 2.0 * self.outer_pad) / self.scale
        height = (self.HEIGHT_FULL - 2.0 * self.outer_pad) / self.scale
        
        if self.grid_type == "GRID":
            grid_width = (width - (self.n_x - 1) * self.grid_pad) / self.n_x
            grid_height = (height - (self.n_y - 1) * self.grid_pad) / self.n_y
            grid_dist_x = grid_width + self.grid_pad
            grid_dist_y = grid_height + self.grid_pad
            for y in range(self.n_y):
                with vsk.pushMatrix():
                    for x in range(self.n_x):
                        self.resample_module_choices()
                        self.update_module_params()
                        
                        generator = StationGenerator(grid_width, grid_height, self.module_types_curr, self.module_type_probs_curr,
                                                     self.probs_modules_parallel, self.probs_modules_normal,
                                                     self.prob_connector_parallel_match_height, 
                                                     weight_continue_same_dir=self.weight_continue_same_dir)
                        generator.generate(num_tries=self.num_tries, num_consec_fails_max=self.num_consec_fails_max)
                        generator.draw(vsk, draw_modules=self.draw_modules, draw_floating_connectors=self.draw_floating_connectors,
                                       debug=self.debug)
                        
                        vsk.translate(grid_dist_x, 0)    
                vsk.translate(0, -grid_dist_y)
        elif self.grid_type == "QUAD":
            tree = QuadTree(self, width, height, 0.0, self.grid_pad, layers=self.quad_layers)
            for node in tree.root.breadth_first():
                if node.bottom:
                    with vsk.pushMatrix():
                        vsk.translate(node.x, node.y)
                        
                        self.resample_module_choices()
                        self.update_module_params()
                        
                        generator = StationGenerator(node.width, node.height, self.module_types_curr, self.module_type_probs_curr,
                                                     self.probs_modules_parallel, self.probs_modules_normal,
                                                     self.prob_connector_parallel_match_height, 
                                                     weight_continue_same_dir=self.weight_continue_same_dir)
                        generator.generate(num_tries=self.num_tries, num_consec_fails_max=self.num_consec_fails_max)
                        generator.draw(vsk, draw_modules=self.draw_modules, draw_floating_connectors=self.draw_floating_connectors,
                                       debug=self.debug)
        
        if self.occult:
            vsk.vpype("occult -i")
            
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    SpacestationSketch.display()
