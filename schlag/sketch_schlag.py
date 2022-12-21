import vsketch
import numpy as np
from numpy.random import default_rng
# from shapely.geometry import Polygon


def draw_filled_circle(vsk, x, y, radius, line_width=1e-2):
    N = int(radius / line_width)
    for r in np.linspace(radius, 0, N):
        vsk.circle(x, y, radius=r)
        
        
def draw_shaded_circle(vsk, x_0, y_0, radius, fill_distance, angle=0.0):
    vsk.circle(x_0, y_0, radius=radius)
    N = np.max((0, int(np.round(2 * (radius / fill_distance - 1)))))
    fill_distance = 2 * radius / (N + 1)
    with vsk.pushMatrix():
        vsk.translate(x_0, y_0)
        vsk.rotate(angle)
        for d in np.linspace(-radius + fill_distance, radius - fill_distance, N, endpoint=True):
            dy = radius * np.sin(np.arccos(d / radius))
            vsk.line(d, -dy, d, dy) 
            

def draw_shaded_rect(vsk, x, y, width, height, fill_distance, angle=0.0):
    vsk.rect(x, y, width, height, mode="center")
    # (N + 2) * fill_distance = width
    N = np.max((0, int(np.round(width / fill_distance - 1))))
    fill_distance = width / (N + 1)
    with vsk.pushMatrix():
        vsk.translate(x, y)
        for x in np.linspace(-0.5*width + fill_distance, 0.5*width - fill_distance, N, endpoint=True):
            vsk.line(x, -0.5*height, x, 0.5*height)
    # TODO: add arbitrary shading angle 

    
def draw_dotted_circle(vsk, x, y, radius, radius_inner):
    vsk.circle(x, y, radius=radius)
    draw_filled_circle(vsk, x, y, radius=radius_inner)
    

class SchlagSketch(vsketch.SketchClass):
    N_grid_x = vsketch.Param(30, min_value=1)
    N_grid_y = vsketch.Param(30, min_value=1)
    width = vsketch.Param(10.0)
    scale = vsketch.Param(1.0, min_value=0.0)
    debug_blob = vsketch.Param(True)
    debug_shapes = vsketch.Param(True)
    occult = vsketch.Param(False)
    padding = vsketch.Param(0.20)
    
    # Metaballs:
    N_metaballs = vsketch.Param(10, min_value=0)
    r_metaballs_max = vsketch.Param(1.2, min_value=0.0)
    r_metaballs_min = vsketch.Param(0.3, min_value=0.0)
    metaballs_thresh = vsketch.Param(1.0, min_value=0.0)
    
    # Rects:
    N_rects_min = vsketch.Param(1, min_value=0)
    N_rects_max = vsketch.Param(5, min_value=0)
    rects_width_min = vsketch.Param(0.4, min_value=0)
    rects_width_max = vsketch.Param(2.0, min_value=0)
    rects_height_gain_max = vsketch.Param(4.0, min_value=0)
    
    def add_rect_to_grid(self, grid, x_0, y_0, width, height, angle, unit_dist):
        N_x = int(np.ceil(1.1*np.sqrt(2) * width / unit_dist))
        N_y = int(np.ceil(1.1*np.sqrt(2) * height / unit_dist))
        for x in np.linspace(-0.5*width, 0.5*width, N_x):
            for y in np.linspace(-0.5*height, 0.5*height, N_y):
                x_paper = x_0 + x * np.cos(angle) + y * np.sin(angle)
                y_paper = y_0 + x * np.sin(angle) - y * np.cos(angle)
                x_index = int(np.round((x_paper) / unit_dist))
                y_index = int(np.round((y_paper) / unit_dist))
                if (0 <= x_index < grid.shape[0]) and (0 <= y_index < grid.shape[1]):
                    grid[x_index,y_index] = True
        return grid

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale)
        
        self.unit_size = self.width / self.N_grid_x
        self.height = self.N_grid_y * self.unit_size
        self.xs_grid = np.linspace(0, self.width, num=self.N_grid_x)
        self.ys_grid = np.linspace(0, self.height, num=self.N_grid_y)
        
        self.rng = default_rng()
        xy_metaballs_min = np.full(2, self.r_metaballs_max) + self.padding
        xy_metaballs_max = np.array([self.width, self.height]) - xy_metaballs_min
        xy_metaballs = self.rng.uniform(low=xy_metaballs_min, high=xy_metaballs_max, size=(self.N_metaballs,2))
        r_metaballs = self.rng.uniform(low=self.r_metaballs_min, high=self.r_metaballs_max, size=self.N_metaballs)
        
        
        # Compute f:
        f_grid = np.zeros((self.N_grid_x, self.N_grid_y))
        occupancy_grid = np.full((self.N_grid_x, self.N_grid_y), False)
        for i, x in enumerate(self.xs_grid):
            for j, y in enumerate(self.ys_grid):
                d = np.array([x,y]) - xy_metaballs
                f = np.sum(r_metaballs**2 / (d[:,0]**2 + d[:,1]**2 + 1e-6))
                f_grid[i,j] = f
                
                if f_grid[i,j] > self.metaballs_thresh:
                    occupancy_grid[i,j] = True
                    
        x_valid, y_valid = np.where(occupancy_grid)
        N_valid = x_valid.shape[0]        
        
        
        N_rects = self.rng.integers(self.N_rects_min, self.N_rects_max + 1)
        vsk.stroke(2)
        for i in range(N_rects):
            choice_index = self.rng.integers(0, N_valid)    
            x_rect, y_rect = self.xs_grid[x_valid[choice_index]], self.ys_grid[y_valid[choice_index]]
            
            rect_angle = self.rng.uniform(0, np.pi)
            rect_width = self.rng.uniform(self.rects_width_min, self.rects_width_max)
            rect_height = rect_width * self.rng.uniform(1.0, self.rects_height_gain_max)  # random orientation so can do this to simplify tuning
            self.add_rect_to_grid(occupancy_grid, x_rect, y_rect, rect_width, rect_height, rect_angle, unit_dist=self.unit_size)
            if self.debug_blob:
                with vsk.pushMatrix():
                    vsk.translate(x_rect, y_rect)
                    vsk.rotate(rect_angle)
                    vsk.rect(0, 0, rect_width, rect_height, mode="center")
        vsk.stroke(1)

        
        # Debug draws:
        if self.debug_blob:
            vsk.stroke(2)
            for xy, r in zip(xy_metaballs, r_metaballs):
                vsk.circle(xy[0], xy[1], radius=r)
            
            vsk.stroke(3)
            for i, x in enumerate(self.xs_grid):
                for j, y in enumerate(self.ys_grid):
                    vsk.circle(x, y, radius=0.01)
                    if occupancy_grid[i,j]:
                        vsk.circle(x, y, radius=0.5*self.unit_size)
            vsk.stroke(1)
                
        if self.debug_shapes:
            draw_shaded_circle(vsk, 0, 0, radius=0.375, fill_distance=0.1, angle=np.deg2rad(45))     
            draw_shaded_rect(vsk, 1, 0, width=0.7, height=0.5, fill_distance=0.1)
            draw_dotted_circle(vsk, 2, 0, radius=0.375, radius_inner=0.2)
            draw_filled_circle(vsk, 3, 0, radius=0.375)
            
            
            
            
            
            
        # xs = np.linspace(0, 1, 10)
        # ys = np.linspace(0, 1, 10)
        # noise_grid = vsk.noise(xs, ys)
        # for x, noise_row in enumerate(noise_grid):
        #     for y, noise in enumerate(noise_row):
        #         if noise > 0.5:
        #             vsk.circle(x, y, radius=0.05)
                    
        if self.occult:
            vsk.vpype("occult -i")
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    SchlagSketch.display()
