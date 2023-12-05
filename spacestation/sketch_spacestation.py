import vsketch
import numpy as np
from numpy.random import default_rng
from enum import Enum
import shapely as sh
import random
from shapely import Polygon, MultiPolygon, Point, MultiPoint, LineString
from plotter_shapes.plotter_shapes import get_empty_sketch
from plotter_util.plotter_util import pick_random_element


def normalize_vec_to_sum_one(vec):
    return vec / np.sum(vec)


def get_points_iterable(points):
    if points.is_empty:
        return []
    elif points.geom_type == "MultiPoint":
        return points.geoms
    elif points.geom_type == "Point":
        return [points]


# Direction = Enum("Direction", "RIGHT UP LEFT DOWN")
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
 

class Structure:
    def __init__(self, x, y, width, height, direction, allow_open_points=True, allow_all_dirs=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.direction = direction
        dir_unit_vec = direction_to_unit_vector(self.direction)
        self.x_center, self.y_center = 0.5 * np.array([self.width, self.height]) * dir_unit_vec + \
                                       np.array([self.x, self.y])

        self.allow_open_points = allow_open_points
        self.allow_all_dirs = allow_all_dirs
        self.open_points = self.init_open_points(allow_all_dirs=self.allow_all_dirs) if self.allow_open_points else None
        
    @classmethod
    def sample_bb_lengths(cls, dir, rng):
        """Default sampling of bounding box size"""
        height = rng.uniform(cls.height_min, cls.height_max)
        width = height * rng.uniform(cls.width_gain_min, cls.width_gain_max)
        
        if dir in (Direction.RIGHT, Direction.LEFT):
            length_x, length_y = width, height
        else:
            length_x, length_y = height, width
            
        return length_x, length_y
    
    @classmethod
    def update(cls, height_min, height_max, width_gain_min, width_gain_max):
        """Default update of class variables"""
        cls.height_min = height_min
        cls.height_max = height_max
        cls.width_gain_min = width_gain_min
        cls.width_gain_max = width_gain_max
    
    def get_bounding_box(self, shrink=0.0):
        assert shrink < 0.5 * min((self.width, self.height))
        return sh.box(self.x_center - 0.5 * self.width + shrink, self.y_center - 0.5 * self.height + shrink,
                      self.x_center + 0.5 * self.width - shrink, self.y_center + 0.5 * self.height - shrink)
    
    def init_open_points(self, dist=2e-1, allow_all_dirs=False):
        sides = {}
        if self.direction != Direction.RIGHT or allow_all_dirs:
            sides[Direction.LEFT] = MultiPoint([(self.x_center - 0.5 * self.width, self.y_center)])
        if self.direction != Direction.UP or allow_all_dirs:
            sides[Direction.DOWN] = MultiPoint([(self.x_center, self.y_center + 0.5 * self.height)])
        if self.direction != Direction.LEFT or allow_all_dirs:
            sides[Direction.RIGHT] = MultiPoint([(self.x_center + 0.5 * self.width, self.y_center)])
        if self.direction != Direction.DOWN or allow_all_dirs:
            sides[Direction.UP] = MultiPoint([(self.x_center, self.y_center - 0.5 * self.height)])
        
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
            return LineString([[self.x, self.y - 0.5 * self.height], 
                               [self.x, self.y + 0.5 * self.height]])
        else:
            return LineString([[self.x - 0.5 * self.width, self.y], 
                               [self.x + 0.5 * self.width, self.y]])
    
    def draw_bounding_box(self):
        sketch = get_empty_sketch()
        sketch.rect(self.x_center, self.y_center, self.width, self.height, mode="center")
        sketch.circle(self.x, self.y, radius=3e-2)
        sketch.circle(self.x_center, self.y_center, radius=3e-2)
        return sketch


class Capsule(Structure):
    def __init__(self, x, y, width, height, direction):
        super().__init__(x, y, width, height, direction)
                
    def draw(self):
        return None
    

class DockingBay(Structure):
    def __init__(self, x, y, width, height, direction):
        super().__init__(x, y, width, height, direction, allow_open_points=False)
        
    def draw(self):
        return None
    
    
class SolarPanel(Structure):
    def __init__(self, x, y, width, height, direction):
        super().__init__(x, y, width, height, direction, allow_open_points=False)
        
    def draw(self):
        return None
        

class StructureGenerator:
    def __init__(self, width, height, prob_structures, weight_continue_same_dir=1.0):
        self.width = width
        self.height = height
        self.weight_continue_same_dir = weight_continue_same_dir
        
        self.rng = default_rng()
        
        self.structures = []
        self.bounding_geometry = Point()  # empty geometry
        
        self.structure_types = [Capsule, DockingBay, SolarPanel]
        self.prob_structures = prob_structures
    
    def get_bounding_box(self):
        return sh.box(-0.5 * self.width, -0.5 * self.height, 0.5 * self.width, 0.5 * self.height)
        
    def add_structure(self, structure, prev_structure=None):
        self.structures.append(structure)
        
        # Add bounding box of new structure to overall bounding geometry:
        bb = structure.get_bounding_box()
        self.bounding_geometry = sh.union(self.bounding_geometry, bb)
        
        # Remove no longer open points from previous structure:
        # TODO: optional margin around the BB when removing points
        if prev_structure is not None:
            line = structure.get_edge_line()
            prev_structure.open_points[structure.direction] = sh.difference(prev_structure.open_points[structure.direction], line)
            if prev_structure.open_points[structure.direction].is_empty:  # remove side if all points are removed
                prev_structure.open_points.pop(structure.direction, None)
    
    def get_open_sides(self):
        sides, weights = [], []
        for idx, structure in enumerate(self.structures):
            if structure.allow_open_points:
                for dir in structure.open_points.keys():
                    if dir == structure.direction:
                        weights.append(self.weight_continue_same_dir)
                    else:
                        weights.append(1.0)
                    sides.append((idx, dir))
        return sides, weights
    
    def generate(self, num_tries, num_consec_fails_max=50):
        x, y, = 0.0, 0.0  # start in zero
        prev_structure = None
        consec_fails = 0
        for i in range(num_tries):
            if consec_fails >= num_consec_fails_max:  # number of consecutive fails termination criteria
                break
            
            if i > 0:
                # Sample random (but weighted) side:
                sides, side_weights = self.get_open_sides()
                if len(side_weights) == 0:  # exit if no options left
                    break
                
                side_probs = normalize_vec_to_sum_one(side_weights)
                idx, dir = pick_random_element(sides, side_probs)
                
                prev_structure = self.structures[idx]
                point = random.choice(get_points_iterable(prev_structure.open_points[dir]))
                x, y = point.x, point.y
                
                structure_class = pick_random_element(self.structure_types, self.prob_structures)
            else:
                structure_class = Capsule  # first placed structure must be capsule
                dir = random.choice(list(Direction))
                
            # Pick random width and height:
            length_x, length_y = structure_class.sample_bb_lengths(dir, self.rng)
            
            structure = structure_class(x, y, length_x, length_y, dir)
            
            # Check if structure fits in bounding geometry:
            intersects_bounding_geom = self.bounding_geometry.intersects(structure.get_bounding_box(shrink=1e-4))
            
            # Check if structure fits in outer bounding box:
            inside_outer_bb = self.get_bounding_box().contains(structure.get_bounding_box())
            
            # TODO: check if it fits on edge line of prev_structure?
            
            if inside_outer_bb and not intersects_bounding_geom:
                consec_fails = 0
                self.add_structure(structure, prev_structure=prev_structure)
            else:
                consec_fails += 1

    def draw_outer_bounding_box(self, vsk):
        vsk.stroke(2)
        vsk.rect(0, 0, self.width, self.height, mode="center")
        vsk.stroke(1)
        
    def draw_bounding_boxes(self, vsk, arrow_length=3e-1, text_offset=0.2):
        vsk.stroke(2)
        for idx, structure in enumerate(self.structures):
            vsk.sketch(structure.draw_bounding_box())
            
            text_offset_curr = text_offset if structure.direction == Direction.UP else -text_offset
            vsk.text(f"{idx}.{type(structure).__name__[:2]}", structure.x_center, 
                     structure.y_center + text_offset_curr, size=0.15, align="center")
            
            with vsk.pushMatrix():  # draw arrow to show direction
                angle = direction_to_angle(structure.direction)
                vsk.translate(structure.x_center, structure.y_center)
                vsk.rotate(-angle)
                vsk.line(0, 0, arrow_length, 0.0)
                vsk.line(arrow_length, 0.0, arrow_length - 0.3 * arrow_length / np.sqrt(2),
                         0.3 * arrow_length / np.sqrt(2))
                vsk.line(arrow_length, 0.0, arrow_length - 0.3 * arrow_length / np.sqrt(2),
                         -0.3 * arrow_length / np.sqrt(2))
        vsk.stroke(1)
            
    def draw_open_points(self, vsk):
        vsk.stroke(3)
        for structure in self.structures:
            if structure.allow_open_points:
                for points in structure.open_points.values():
                    for point in get_points_iterable(points):
                        vsk.circle(point.x, point.y, radius=2e-2)
        vsk.stroke(1)
        
    def draw(self, vsk):
        pass
            

class SpacestationSketch(vsketch.SketchClass):
    WIDTH_FULL = 21
    HEIGHT_FULL = 29.7
    
    debug = vsketch.Param(True)
    occult = vsketch.Param(False)
    scale = vsketch.Param(1.0)
    
    num_tries = vsketch.Param(20, min_value=1)
    num_consec_fails_max = vsketch.Param(50, min_value=1)
    
    n_x = vsketch.Param(1, min_value=1)
    n_y = vsketch.Param(1, min_value=1)
    grid_dist_x = vsketch.Param(8.0)
    grid_dist_y = vsketch.Param(8.0)
    
    weight_continue_same_dir = vsketch.Param(6.0, min_value=0.0)
    
    prob_capsule = vsketch.Param(0.7, min_value=0.0, max_value=1.0)
    prob_docking_bay = vsketch.Param(0.1, min_value=0.0, max_value=1.0)
    prob_solar_panel = vsketch.Param(0.2, min_value=0.0, max_value=1.0)
        
    capsule_height_min = vsketch.Param(1.0, min_value=0)
    capsule_height_max = vsketch.Param(2.0, min_value=0)
    capsule_width_gain_min = vsketch.Param(1.0, min_value=0)
    capsule_width_gain_max = vsketch.Param(3.0, min_value=0)
    
    dock_height_min = vsketch.Param(1.0, min_value=0)
    dock_height_max = vsketch.Param(2.0, min_value=0)
    dock_width_gain_min = vsketch.Param(0.1, min_value=0)
    dock_width_gain_max = vsketch.Param(0.2, min_value=0)
    
    solar_height_min = vsketch.Param(0.8, min_value=0)
    solar_height_max = vsketch.Param(1.6, min_value=0)
    solar_width_gain_min = vsketch.Param(4.0, min_value=0)
    solar_width_gain_max = vsketch.Param(8.0, min_value=0)

    def init_drawing(self, vsk):
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale, self.scale)
        print("\n")
        
    def init_probs(self):
        probs = np.array([self.prob_capsule, self.prob_docking_bay, self.prob_solar_panel])
        self.prob_structures = normalize_vec_to_sum_one(probs)
    
    def init_structures(self):
        Capsule.update(self.capsule_height_min, self.capsule_height_max, self.capsule_width_gain_min,
                       self.capsule_width_gain_max)
        DockingBay.update(self.dock_height_min, self.dock_height_max, self.dock_width_gain_min,
                          self.dock_width_gain_max)
        SolarPanel.update(self.solar_height_min, self.solar_height_max, self.solar_width_gain_min,
                          self.solar_width_gain_max)
        
    def draw(self, vsk: vsketch.Vsketch) -> None:
        self.init_probs()
        self.init_structures()
        self.init_drawing(vsk)
        
        width = 20.0
        height = 28.5
        generator = StructureGenerator(width, height, self.prob_structures, 
                                       weight_continue_same_dir=self.weight_continue_same_dir)
        generator.generate(num_tries=self.num_tries, num_consec_fails_max=self.num_consec_fails_max)
        
        if self.debug:
            vsk.circle(0, 0, radius=1e-1)  # origin
            generator.draw_outer_bounding_box(vsk)
            generator.draw_bounding_boxes(vsk)
            generator.draw_open_points(vsk)

        if self.occult:
            vsk.vpype("occult -i")
            
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    SpacestationSketch.display()
