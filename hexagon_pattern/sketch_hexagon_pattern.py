import vsketch
import numpy as np
# import plotter_util as plutil
# from numpy.random import uniform, randint

# TODO: make a cool border. make shaded sides with line density param.

class HexagonPatternSketch(vsketch.SketchClass):
    # Sketch parameters:
    
    angle = vsketch.Param(30.0, step=1.0)
    size = 1.0
    
    n_y = vsketch.Param(1, min_value=1)
    n_x = vsketch.Param(1, min_value=1)
    
    width = 21
    height = 29.7

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        angle_rad = np.deg2rad(self.angle)
        
        for y in range(self.n_y):
            for x in range(self.n_x):
                vsk.line(0, 0, self.size * np.cos(angle_rad), self.size * np.sin(angle_rad))
                vsk.line(self.size * np.cos(angle_rad), self.size * np.sin(angle_rad), 2 * self.size * np.cos(angle_rad), 0)
                vsk.line(0, 0, self.size * np.cos(angle_rad), -self.size * np.sin(angle_rad))
                vsk.translate(self.size * np.cos(angle_rad), -self.size * np.sin(angle_rad))
                vsk.line(0, 0, 0, -self.size)
                vsk.line(0, 0, self.size * np.cos(angle_rad), self.size * np.sin(angle_rad))
                vsk.translate(self.size * np.cos(angle_rad), self.size * np.sin(angle_rad))
                vsk.line(0, 0, 0, -self.size)
            parity_offset = (y % 2) - 0.5
            vsk.translate(-2 * (self.n_x + parity_offset) * self.size * np.cos(angle_rad), self.size + self.size * np.sin(angle_rad))
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    HexagonPatternSketch.display()
