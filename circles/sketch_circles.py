import vsketch
import numpy as np
import numpy.random as random
import scipy.stats as stats

def get_truncated_normal(mean=0, std=1, lower=0, upper=10):
    return stats.truncnorm(
        (lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs()

class CirclesSketch(vsketch.SketchClass):
    # Sketch parameters:
    N_circles = vsketch.Param(16, min_value=1)
    r_mean = vsketch.Param(0.5, min_value=0)
    r_std = vsketch.Param(0.5, min_value=0)
    r_step_mean = vsketch.Param(0.2, min_value=0)
    r_step_std = vsketch.Param(0.5, min_value=0)
    min_steps = vsketch.Param(8, min_value=0)
    max_steps = vsketch.Param(16, min_value=0)
    width = 21
    height = 29.7
    padding = vsketch.Param(3.0, min_value=0)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        for i in range(self.N_circles):
            radius = get_truncated_normal(self.r_mean, self.r_std, 0.01, 10.0)
            x = np.random.uniform(self.padding, self.width - self.padding)
            y = np.random.uniform(self.padding, self.height - self.padding)
            n = np.random.randint(self.min_steps, self.max_steps)
            k = get_truncated_normal(self.r_step_mean, self.r_step_std, 8e-2, 0.5)
            for j in range(n):
                vsk.circle(x, y, radius + k*j, mode="radius")

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    CirclesSketch.display()
