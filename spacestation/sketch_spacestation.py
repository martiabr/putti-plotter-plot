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


def normalize_mat_to_row_sum_one(mat):
    return mat / np.sum(mat, axis=-1)[:,None]


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
        sketch.rect(self.x_center, self.y_center, self.length_x, self.length_y, mode="center")
        sketch.circle(self.x, self.y, radius=3e-2)
        sketch.circle(self.x_center, self.y_center, radius=3e-2)
        return sketch


class Capsule(Module):
    def __init__(self, x, y, width, height, direction, from_module, allow_all_dirs=False):
        super().__init__(x, y, width, height, direction, allow_all_dirs=allow_all_dirs)

    @classmethod
    def sample_bb_dims(cls, rng, from_height):
        return super(Capsule, cls).sample_bb_dims(rng, from_height, match_from_height=True)
    
    def draw(self):
        sketch = get_empty_sketch()
        sketch.rect(self.x_center, self.y_center, self.length_x, self.length_y, mode="center")
        return sketch


class CapsuleVariation1(Capsule):
    def draw(self):
        sketch = super().draw()
        return sketch
    

class Connector(Module):
    def __init__(self, x, y, width, height, direction, from_module):
        # width and height have been sampled to match from module.
        # from height is saved already
        # to height must be set somewhere...
        # also sample_bb_dims is a class method and gets us the width and height. 
        # Meaning here we need to figure out to_height and update width, height, length_x and length_y if it is changed.
        # Not the nicest but works out ok.
        # Need to 1. get height as a gain of from height, while 2. clamping to max/min height values

        # Determine end-module height (for drawing), and update height accordingly to get correct BB size:
        dirs_normal = directions_are_normal(direction, from_module.direction)
        self.from_height = from_module.width if dirs_normal else from_module.height
        if dirs_normal:
            end_height = np.random.uniform(Connector.height_min, Connector.height_max)
        else:
            end_height = self.from_height * np.random.uniform(Connector.from_height_gain_min, Connector.from_height_gain_max)
        self.end_height = np.clip(end_height, Connector.height_min, Connector.height_max)
        height = np.max((height, self.end_height))
        
        # Problem: connectors can be added in all directions of capsule. 
        # On the width side we dont want to match the width, just sample uniform between max and min connector height, but with limiting to width of the capsule.
        # On the height side we want some prob of matching height and some prob of just doing like width, except limiting to height of capsule.
        # This requires some more restructuring:
        # When calling sample_bb_dims we do the checks and input from_height and bool vals to match what is written above.
        # ALso here we need to update the from_height, end_height and height slightly?
        
        super().__init__(x, y, width, height, direction)
        self.open_points = dict(zip([self.direction], [self.open_points[self.direction]]))  # connector type can only build forward
        
    # @classmethod
    # def sample_bb_dims(cls, rng, from_height, limit_height_by_from_height=False, match_from_height=False):
    #     return super(Connector, cls).sample_bb_dims(rng, from_height, limit_height_by_from_height=limit_height_by_from_height,
    #                                                 match_from_height=match_from_height)
    
    @classmethod
    def update(cls, height_min, height_max, from_height_gain_min, from_height_gain_max, width_gain_min, width_gain_max):
        """Default update of class variables"""
        cls.height_min = height_min
        cls.height_max = height_max
        cls.from_height_gain_min = from_height_gain_min
        cls.from_height_gain_max = from_height_gain_max
        cls.width_gain_min = width_gain_min
        cls.width_gain_max = width_gain_max
         
    def draw(self):
        sketch = get_empty_sketch()
        return sketch
    
    
class ConnectorVariation1(Connector):
    def draw(self):
        sketch = get_empty_sketch()
        return sketch
    
    
class SolarPanel(Module):
    def __init__(self, x, y, width, height, direction):
        super().__init__(x, y, width, height, direction)
        self.open_points = None  # dont build outwards from this module


class SolarPanelSingle(SolarPanel):
    def draw(self):
        sketch = get_empty_sketch()
        return sketch
    

class SolarPanelDouble(SolarPanel):
    def draw(self):
        sketch = get_empty_sketch()
        return sketch
    

class Decoration(Module):
    def __init__(self, x, y, width, height, direction):
        super().__init__(x, y, width, height, direction)
        self.open_points = None  # dont build outwards from this module
    
    @classmethod
    def sample_bb_dims(cls, rng, from_height):
        return super(Decoration, cls).sample_bb_dims(rng, from_height, limit_height_by_from_height=True)
    

class DockingBay(Decoration):
    def draw(self):
        sketch = get_empty_sketch()
        return sketch
    
    
class StationGenerator:
    def __init__(self, width, height, module_types, probs_modules_parallel, probs_modules_normal, prob_connector_parallel_match_height,
                 weight_continue_same_dir=1.0):
        self.width = width
        self.height = height
        
        self.rng = default_rng()
        
        self.modules = []
        self.bounding_geometry = Point()  # empty geometry
        
        self.module_types = module_types
        self.n_main_module_types = len(self.module_types)
        self.module_type_to_idx = dict(zip(self.module_types.keys(), range(self.n_main_module_types)))
        
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
    
    def pick_random_module(self, from_module, dir):
        # TODO: update me to work with submodules
        from_module_idx = self.module_type_to_idx[type(from_module)]
        if directions_are_normal(dir, from_module.direction):
            probs = self.probs_modules_normal[from_module_idx]
        else:
            probs = self.probs_modules_parallel[from_module_idx]
        return pick_random_element(self.module_types.keys(), probs)
    
    def generate(self, num_tries, num_consec_fails_max=50):
        x, y, = 0.0, 0.0  # start in zero
        from_module = None
        consec_fails = 0
        for i in range(num_tries):
            if consec_fails >= num_consec_fails_max:  # number of consecutive fails termination criteria
                print("Termination: number of consecutive fails reached.")
                break
            
            if i == 0:
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
                
                module_class = self.pick_random_module(from_module, dir)
                # TODO: update pick random module to pick submodule as well
                
                # Pick random width and height:
                # TODO: if this gets more complicated build args dict and input **args instead
                if issubclass(module_class, Capsule) and isinstance(from_module, Connector):
                    width, height = module_class.sample_bb_dims(self.rng, from_height=from_module.end_height)
                elif issubclass(module_class, Connector):
                    are_normal = directions_are_normal(dir, from_module.dir)
                    if are_normal:
                        width, height = module_class.sample_bb_dims(self.rng, from_module.width, limit_height_by_from_height=True)
                    elif self.rng.random() < self.prob_connector_parallel_match_height:
                        width, height = module_class.sample_bb_dims(self.rng, from_module.height, match_from_height=True)
                    else:
                        width, height = module_class.sample_bb_dims(self.rng, from_module.height, limit_height_by_from_height=True)
                else:
                    width, height = module_class.sample_bb_dims(self.rng, from_module.height)
                module = module_class(x, y, width, height, dir)
            else:
                # TODO: pick random Capsule submodule!
                module_class = Capsule  # first placed module must be capsule
                dir = random.choice(list(Direction))
                
                width, height = module_class.sample_bb_dims(self.rng, from_height=None)
                module = module_class(x, y, width, height, dir, allow_all_dirs=True)
        
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
        vsk.stroke(3)
        for module in self.modules:
            if module.open_points is not None:
                for points in module.open_points.values():
                    for point in get_points_iterable(points):
                        vsk.circle(point.x, point.y, radius=2e-2)
        vsk.stroke(1)
        
    def draw(self, vsk):
        for module in self.modules:
            vsk.sketch(module.draw())
            

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
    
    prob_connector_parallel_match_height = vsketch.Param(0.7, min_value=0.0, max_value=1.0)
    
    prob_capsule_capsule_parallel = vsketch.Param(1.0, min_value=0)
    prob_capsule_connector_parallel = vsketch.Param(2.0, min_value=0)
    prob_capsule_solar_parallel = vsketch.Param(0.4, min_value=0)
    prob_capsule_dock_parallel = vsketch.Param(1.0, min_value=0)

    prob_capsule_capsule_normal = vsketch.Param(1.0, min_value=0)
    prob_capsule_connector_normal = vsketch.Param(3.0, min_value=0)
    prob_capsule_solar_normal = vsketch.Param(1.0, min_value=0)
    prob_capsule_dock_normal = vsketch.Param(1.0, min_value=0)
        
    capsule_height_min = vsketch.Param(1.0, min_value=0)
    capsule_height_max = vsketch.Param(2.0, min_value=0)
    capsule_width_gain_min = vsketch.Param(1.0, min_value=0)
    capsule_width_gain_max = vsketch.Param(3.0, min_value=0)
    
    connector_height_min = vsketch.Param(1.0, min_value=0)
    connector_height_max = vsketch.Param(2.0, min_value=0)
    connector_from_height_gain_min = vsketch.Param(0.833, min_value=0)
    connector_from_height_gain_max = vsketch.Param(1.2, min_value=0)
    connector_width_gain_min = vsketch.Param(0.2, min_value=0)
    connector_width_gain_max = vsketch.Param(0.4, min_value=0)
    
    solar_height_min = vsketch.Param(0.8, min_value=0)
    solar_height_max = vsketch.Param(1.6, min_value=0)
    solar_width_gain_min = vsketch.Param(4.0, min_value=0)
    solar_width_gain_max = vsketch.Param(8.0, min_value=0)
    
    dock_height_min = vsketch.Param(1.0, min_value=0)
    dock_height_max = vsketch.Param(2.0, min_value=0)
    dock_width_gain_min = vsketch.Param(0.1, min_value=0)
    dock_width_gain_max = vsketch.Param(0.2, min_value=0)
    

    def init_drawing(self, vsk):
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale, self.scale)
        print("\nRunning...")
        
    def init_probs(self):
        # probs = np.array([self.prob_capsule, self.prob_connector, self.prob_solar_panel, self.prob_docking_bay])
        # self.prob_modules = normalize_vec_to_sum_one(probs)
        probs_parallel = np.array([[self.prob_capsule_capsule_parallel, self.prob_capsule_connector_parallel,
                                    self.prob_capsule_solar_parallel, self.prob_capsule_dock_parallel],
                                   [1.0, 0.0, 0.0, 0.0],
                                   4 * [np.nan],
                                   4 * [np.nan]])
        self.probs_modules_parallel = normalize_mat_to_row_sum_one(probs_parallel)
        
        probs_normal = np.array([[self.prob_capsule_capsule_normal, self.prob_capsule_connector_normal,
                                  self.prob_capsule_solar_normal, self.prob_capsule_dock_normal],
                                 4 * [np.nan],
                                 4 * [np.nan],
                                 4 * [np.nan]])
        self.probs_modules_normal = normalize_mat_to_row_sum_one(probs_normal)
    
    def init_modules(self):
        Capsule.update(self.capsule_height_min, self.capsule_height_max, self.capsule_width_gain_min,
                       self.capsule_width_gain_max)
        Connector.update(self.connector_height_min, self.connector_height_max, self.connector_from_height_gain_min,
                         self.connector_from_height_gain_max, self.connector_width_gain_min, self.connector_width_gain_max)
        SolarPanel.update(self.solar_height_min, self.solar_height_max, self.solar_width_gain_min,
                          self.solar_width_gain_max)
        Decoration.update(self.dock_height_min, self.dock_height_max, self.dock_width_gain_min,
                          self.dock_width_gain_max)
        
    def draw(self, vsk: vsketch.Vsketch) -> None:
        self.init_probs()
        self.init_modules()
        self.init_drawing(vsk)
        
        width = 20.0
        height = 28.5
        
        # module_types = [Capsule, Connector, SolarPanel, Decoration]
        module_types = {Capsule: [CapsuleVariation1],
                        Connector: [ConnectorVariation1],
                        SolarPanel: [SolarPanelSingle, SolarPanelDouble],
                        Decoration: [DockingBay]}
        
        generator = StationGenerator(width, height, module_types, self.probs_modules_parallel, 
                                     self.probs_modules_normal, self.prob_connector_parallel_match_height, weight_continue_same_dir=self.weight_continue_same_dir)
        generator.generate(num_tries=self.num_tries, num_consec_fails_max=self.num_consec_fails_max)
        generator.draw()
        
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
