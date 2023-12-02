import vsketch
import numpy as np
from numpy.random import default_rng
from enum import Enum
import shapely as sh
from shapely import Polygon, MultiPolygon, Point, MultiPoint, LineString
from plotter_shapes.plotter_shapes import get_empty_sketch


# Look at KSP space stations
# Centrifuge?
# Can variable width line be used at all?
# Remember that vsketch Shapes will be necessary here to create more intricate shapes.
# Here we can also add a shapely geometry directly to the shape. So we can just create things in shapely and add as a shape to draw it.

# Elements in the space stations:
# - Solar panels (single panel directly outwards, double panels directly outwards, or single/double panel on arm)
#   - Different connection types
# - Capsules of different sizes and designs
#   - Antennas, windows, cargo doors
# - Circular inflatable modules?
# - Docking ports
# - Bars like in Freedom 
# - Centrifuge ring (but hard to draw in 2d, either 4 modules or just a ring. Try to get 3d effect with nonuniform lines, should be also to figure out that math)
# - Crew capsules/cabins (capsules with windows)
# - Robot arms, antennas, ...
# - communication dish tower
# - Possibly a spacecraft docked in a bay?
# - Need to look at more future oriented space stations for more ideas

# Separate dungeon generator and the actual drawing

# Somehow have weights to try to force symmetries?

# Algorithm:
# - list of structures
# - a structure has a width and height, which is added to the bounding_geometry to keep track of occupied space.
#       this is used to check if we can place a new structure.
# - when placing a structure we also sample a string of points along the edges where new structures may be placed.
# - when finding the next structure to be placed, we simply pick a random point. 
# - some points may have extra weight to them to get desired behaviour?
# - when placing a new structure we update the bounding geometry with a union and the valid next points with a subtract.
# - each point must also have a direction associated with it to know how to place the next structure.
# - we must also check for collision with outer bounds of the entire space station. This is quite easy to do without having to
#       use shapely. Just check if each edge of the bb is within the bounds. 

# - the most difficult part is how to structure the data with all the open points, and their associated data (direction and weight).
# - e.g. a capsule will only have one open point on the end, but many above and below. Then need to add weight so the point
#       on the end will still have a good chance of being picked. 
# - Are there any alternative ways of doing it?
#       - Only pick the sides and then sample amongst the points. Then weights can be dropped. 
#       - Then when the list of points is empty the side will be removed from consideration.
#       - The main problem is however how we do the intersection to find points which must be removed and then dropping their 
        # data as well. One option is that the xy position is a key in a dict. Works ok but a bit hacky. 
        # thinking about this a bit more, if a structure is valid it will only remove points from the "previous" structure, 
        # i.e. the one it extends out from. This means we can avoid having a multipoint with all possible points
        # instead only the structure itself has a multipoint with its open points which is updated as new structures build out from it.
        # How do we pick new points? We maintain a list of all open sides, on format (idx, dir), so we can then pick a random
        # side and look it up. As the structure is added and open points are removed we also delete the side if it is empty.
        # Afterwards weights could be added if necessary. But weighting each side and each point the same should work ok.


def get_points_iterable(points):
    if points.is_empty:
        return []
    elif points.geom_type == "MultiPoint":
        return points.geoms
    elif points.geom_type == "Point":
        return [points]


Direction = Enum("Direction", "RIGHT UP LEFT DOWN")


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


class Structure:
    def __init__(self, x, y, width, height, direction):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.direction = direction
        unit_vector = direction_to_unit_vector(self.direction)
        self.x_center, self.y_center = 0.5 * np.array([self.width, self.height]) * unit_vector + np.array([self.x, self.y])

        self.open_points = self.init_open_points()

    def get_bounding_box(self):
        return sh.box(self.x_center - 0.5 * self.width, self.y_center - 0.5 * self.height,
                      self.x_center + 0.5 * self.width, self.y_center + 0.5 * self.height)
    
    def init_open_points(self, dist=2e-1):
        # TODO: add margin on ends of linspaces, we dont want to draw stuff all the way to the end...
        n_points_y = int(np.round(self.height / dist))
        points_y = 0.5 * np.linspace(-self.height, self.height, num=n_points_y, endpoint=True)
        
        sides = {}
        if self.direction in (Direction.RIGHT, Direction.LEFT):
            n_points_x = int(np.round(self.width / dist))
            points_x = 0.5 * np.linspace(-self.width, self.width, num=n_points_x, endpoint=True) + self.x_center
            
            points_up = [(x, y) for (x, y) in zip(points_x, n_points_x * [self.y_center - 0.5 * self.height])]
            sides[Direction.UP] = MultiPoint(points_up)
            
            points_down = [(x, y) for (x, y) in zip(points_x, n_points_x * [self.y_center + 0.5 * self.height])]
            sides[Direction.DOWN] = MultiPoint(points_down)
            
            if self.direction == Direction.RIGHT:
                sides[Direction.RIGHT] = MultiPoint([(self.x_center + 0.5 * self.width, self.y_center)])
            else:
                sides[Direction.LEFT] = MultiPoint([(self.x_center - 0.5 * self.width, self.y_center)])
        else:  # direction in UP, DOWN
            n_points_y = int(np.round(self.height / dist))
            points_y = 0.5 * np.linspace(-self.height, self.height, num=n_points_y, endpoint=True) + self.y_center
            
            points_right = [(x, y) for (x, y) in zip(n_points_y * [self.x_center + 0.5 * self.width], points_y)]
            sides[Direction.RIGHT] = MultiPoint(points_right)
            
            points_left = [(x, y) for (x, y) in zip(n_points_y * [self.x_center - 0.5 * self.width], points_y)]
            sides[Direction.LEFT] = MultiPoint(points_left)
            
            if self.direction == Direction.UP:
                sides[Direction.UP] = MultiPoint([(self.x_center, self.y_center - 0.5 * self.height)])
            else:
                sides[Direction.DOWN] = MultiPoint([(self.x_center, self.y_center + 0.5 * self.height)])
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
        

class StructureGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.structures = []
        self.bounding_geometry = []
        # self.open_sides = []
    
    def add_structure(self, structure, prev_structure=None):
        self.structures.append(structure)
        
        # Add bounding box of new structure to overall bounding geometry:
        bb = structure.get_bounding_box()
        self.bounding_geometry = sh.union(self.bounding_geometry, bb)
        
        # # Add open sides of new structure to all open sides: 
        # sides = structure.sample_open_points()
        # self.open_sides.extend(sides)
        
        # Remove no longer open points from previous structure:
        # TODO: optional margin around the BB when removing points
        if prev_structure is not None:
            line = structure.get_edge_line()
            prev_structure.open_points[structure.direction] = sh.difference(prev_structure.open_points[structure.direction], line)
        
    def generate(self, num_tries, num_consec_fails_max=50):
        x, y, = 0.0, 0.0  # start in zero
        structure = Capsule(x, y, 5.0, 2.5, Direction.RIGHT)
        self.add_structure(structure)
        
        x, y = 5.0, 0.0
        structure_2 = Capsule(x, y, 4.0, 2.0, Direction.RIGHT)
        self.add_structure(structure_2, prev_structure=structure)
        
        x, y = 7.0, -1.0
        structure_3 = Capsule(x, y, 2.0, 3.0, Direction.UP)
        self.add_structure(structure_3, prev_structure=structure_2)
        
    def draw_bounding_boxes(self, vsk):
        vsk.stroke(2)
        for structure in self.structures:
            vsk.sketch(structure.draw_bounding_box())
        vsk.stroke(1)
            
    def draw_open_points(self, vsk):
        vsk.stroke(3)
        for structure in self.structures:
            for points in structure.open_points.values():
                for point in get_points_iterable(points):
                    vsk.circle(point.x, point.y, radius=2e-2)
        vsk.stroke(1)
        
    def draw(self, vsk):
        pass
            

class SpacestationSketch(vsketch.SketchClass):
    debug = vsketch.Param(True)
    occult = vsketch.Param(False)
    scale = vsketch.Param(1.0)
    
    n_x = vsketch.Param(4, min_value=1)
    n_y = vsketch.Param(6, min_value=1)
    grid_dist_x = vsketch.Param(8.1)
    grid_dist_y = vsketch.Param(8.4)
    
    WIDTH_FULL = 21
    HEIGHT_FULL = 29.7
    
    rng = default_rng()

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale, self.scale)
        
        generator = StructureGenerator(50.0, 50.0)
        generator.generate(num_tries=2)
        
        if self.debug:
            vsk.circle(0, 0, radius=1e-1)
            generator.draw_bounding_boxes(vsk)
            generator.draw_open_points(vsk)

        if self.occult:
            vsk.vpype("occult -i")
            
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    SpacestationSketch.display()
