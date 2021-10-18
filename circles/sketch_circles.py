import numpy as np
from numpy.random import uniform, randint
import vsketch
import plotter_util as plutil

class CirclesSketch(vsketch.SketchClass):
    # Sketch parameters:
    N_circles = vsketch.Param(16, min_value=1)
    r_mean = vsketch.Param(0.5, min_value=0)
    r_std = vsketch.Param(0.5, min_value=0)
    r_step_mean = vsketch.Param(0.2, min_value=0)
    r_step_std = vsketch.Param(0.5, min_value=0)
    min_steps = vsketch.Param(8, min_value=0)
    max_steps = vsketch.Param(16, min_value=0)
    padding = vsketch.Param(3.0, min_value=0)
    shape_type = vsketch.Param('circle', choices=['circle', 'polygon'])
    polygon_sides = vsketch.Param(3, min_value=3)
    
    width = 21
    height = 29.7

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        for i in range(self.N_circles):
            x = uniform(self.padding, self.width - self.padding)
            y = uniform(self.padding, self.height - self.padding)
            
            radius = plutil.get_truncated_normal(self.r_mean, self.r_std, 0.01, 10.0)
            k = plutil.get_truncated_normal(self.r_step_mean, self.r_step_std, 8e-2, 0.5)
            phase = uniform(-np.pi, np.pi)
            
            if self.min_steps == self.max_steps:
                n = self.min_steps
            else:
                n = randint(self.min_steps, self.max_steps)

            for j in range(n):
                if self.shape_type == 'circle':
                    vsk.circle(x, y, radius + k*j, mode="radius")
                elif self.shape_type == 'polygon':
                    angles = np.linspace(0, 2 * np.pi, self.polygon_sides + 1)
                    vsk.polygon(x + (radius + k*j) * np.cos(angles + phase), y + (radius + k*j) * np.sin(angles + phase))
                    

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")

if __name__ == "__main__":
    CirclesSketch.display()
