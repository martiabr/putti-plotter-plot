import vsketch
import numpy as np

# Ideas:
# - isometric grid with towers, bridges, slopes etc.
# - 3d tubes using circles

# We define grid and box like this:
# \            /
#  \    / \   /
#   \ y \ / x/
#    \   o  /
#     \    /
#    y \  / x
#       \/



class isoShape:
    def __init__(self, x, y, z, x_size, y_size, z_size) -> None:
        self.x_euc, self.y_euc, self.z_euc = x, y, z
        self.x_size, self.y_size, self.z_size = x_size, y_size, z_size
        self.set_iso_from_euc(x, y, z)
        self.set_min_max_values()
        
    def compute_horizontal_dist(self, x_iso, y_iso):
        return (x_iso - y_iso) * np.cos(np.pi / 6)
    
    def compute_vertical_dist(self, x_iso, y_iso):
        return (x_iso + y_iso) * 0.5
        
    def set_iso_from_euc(self, x, y, z):
        self.x_iso = x + z
        self.y_iso = y + z
        self.dist_h = self.compute_horizontal_dist(self.x_iso, self.y_iso)
        self.dist_v = self.compute_vertical_dist(self.x_iso, self.y_iso)
        
    def set_min_max_values(self):
        # TODO: first 4 wrong
        self.x_min = self.x_iso
        self.x_max = self.x_iso + self.x_size
        self.y_min = self.y_iso
        self.y_max = self.y_iso + self.y_size
        self.h_min = self.compute_horizontal_dist(self.x_iso, self.y_iso + self.y_size)
        self.h_max = self.compute_horizontal_dist(self.x_iso + self.x_size, self.y_iso)

def check_iso_shape_overlap(shape_1, shape_2):
    return not (shape_1.x_min >= shape_2.x_max or shape_2.x_min >= shape_1.x_max) and \
           not (shape_1.y_min >= shape_2.y_max or shape_2.y_min >= shape_1.y_max) and \
           not (shape_1.h_min >= shape_2.h_max or shape_2.h_min >= shape_1.h_max)

class IsoSketch(vsketch.SketchClass):

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")

        # implement your sketch here
        # vsk.circle(0, 0, 1.0, mode="radius")
        # vsk.rect(0, 0, 1, 2)
        
        shape_1 = isoShape(0, 0, 0, 1, 1, 2)
        
        vsk.vpype("occult")
        
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    IsoSketch.display()
