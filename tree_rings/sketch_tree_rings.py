import vsketch
import numpy as np
from plotter_util import get_truncated_normal


class TreeRingsSketch(vsketch.SketchClass):
    scale = vsketch.Param(1.0)
    N_segments = vsketch.Param(50)
    N_rings_min = vsketch.Param(8)
    N_rings_max = vsketch.Param(8)
    N_trees = vsketch.Param(25)
    noise_frequency = vsketch.Param(0.5)
    noise_gain = vsketch.Param(0.5)
    min_radius = vsketch.Param(0.7, decimals=2)
    max_radius = vsketch.Param(1.3, decimals=2)
    noise = vsketch.Param('uniform', choices=['uniform_rect', 'uniform_circle', 'gaussian_circle'])
    gaussian_std = vsketch.Param(7e-1)
    
    x_max = vsketch.Param(6.0)
    y_max = vsketch.Param(9.0)

    randomize_start_angle = vsketch.Param(True)
    debug_draws = vsketch.Param(False)
    use_occult = vsketch.Param(True)
    sort_by_radius = vsketch.Param(True)
    grid = vsketch.Param(False)
    grid_x = vsketch.Param(10)
    grid_y = vsketch.Param(10)


    def draw_tree_rings(self, vsk, x_noise_i, y_noise_i, n_rings, radius, randomize_start_angle=True):
        vsk.noiseSeed(np.random.randint(1e6))
        dt = 2 * np.pi / self.N_segments
        with vsk.pushMatrix():
            vsk.translate(x_noise_i, y_noise_i)
            points = np.zeros((self.N_segments,2))
            t_0 = np.random.uniform(0.0, 2 * np.pi) if randomize_start_angle else 0.0
            for i in range(self.N_segments):
                t = t_0 + dt * i
                x_noise_i = self.noise_frequency * np.interp(np.cos(t), (-1, 1), (0, 1))
                y_noise_i = self.noise_frequency * np.interp(np.sin(t), (-1, 1), (0, 1))
                r = radius * (1.0 + self.noise_gain *
                                   np.interp(self.vsk.noise(x_noise_i, y_noise_i),
                                             (0, 1), (-1, 1)))
                x_i = r * np.cos(t)
                y_i = r * np.sin(t)
                points[i] = [x_i, y_i]
                        
            r_start = 1.0 / n_rings
            for r in np.linspace(1.0, r_start, n_rings):
                vsk.polygon(r * points[:,0], r * points[:,1], close=True)
            
            
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        vsk.scale(self.scale, self.scale)

        vsk.noiseDetail(3, falloff=0.5)
        
        if self.grid: self.N_trees = self.grid_x * self.grid_y

        points = np.zeros((self.N_trees,2))
        if self.grid:
            for i, x_i in enumerate(np.linspace(0.0, self.x_max, self.grid_x)):
                for j, y_j in enumerate(np.linspace(0.0, self.y_max, self.grid_y)):
                    points[j + i*self.grid_y] = [x_i, y_j]
        else:
            for i in range(self.N_trees):
                if self.noise == 'uniform_rect':
                    x, y = np.random.uniform([0.0, 0.0], [self.x_max, self.y_max])
                elif self.noise == 'gaussian_circle' or self.noise == 'uniform_circle':
                    if self.noise == 'gaussian_circle':
                        r = get_truncated_normal(0.0, self.gaussian_std, 0.0, 1.0)
                    elif self.noise == 'uniform_circle':
                        r = np.sqrt(np.random.uniform(0.0, 1.0))
                    theta = np.random.normal(0.0, 2 * np.pi)
                    x = self.x_max * r * np.cos(theta)
                    y = self.y_max * r * np.sin(theta)
                points[i] = [x, y]
        
        if self.sort_by_radius:
            distances_squared = points[:,0]**2 + points[:,1]**2
            indices = np.argsort(distances_squared)[::-1]
            points = np.array(points)[indices]
           
        for point in points:
            n_rings = int(np.round(np.random.uniform(self.N_rings_min, self.N_rings_max)))
            radius = np.random.uniform(self.min_radius, self.max_radius)
            self.draw_tree_rings(vsk, point[0], point[1], n_rings, radius,
                                 randomize_start_angle=self.randomize_start_angle)
        
        if self.use_occult:
            vsk.vpype("occult -i")
        
        # Debug draws:
        if self.debug_draws:
            if self.noise == 'uniform_rect':
                vsk.rect(0, 0, self.x_max, self.y_max)
            elif self.noise in ('uniform_circle', 'gaussian_circle'):
                vsk.ellipse(0, 0, 2*self.x_max, 2*self.y_max)
        
        
        # Where to go from here?
        # - Random radius (x)
        # - Speed (line profiler?)
        # - Option for Gaussian noise on position instead of uniform? Should make more of a blob (x)
        # - Or do uniform distribution of position on circle to get more of a blob instead of rectangle (x)
        # - The best would be gaussian distribution, but on a circle (x)
        # - Some sort of sorting by radius in order to get the "blob" look? (x)
        # - Fix radius debug draw (x)
        # - Uniform circle (x)
        # - Instead of random sampled radius just have set radius?
        # - Option for grid instead of random
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    TreeRingsSketch.display()
