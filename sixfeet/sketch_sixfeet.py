import vsketch
import numpy as np
from plotter_shapes.plotter_shapes import *


def get_normal(x, y, x_next, y_next):
    dx, dy = x_next - x, y_next - y
    return np.array([-dy, dx])


def get_unit_normal(x, y, x_next, y_next):
    normal = get_normal(x, y, x_next, y_next)
    return normal / np.sqrt(normal[0]**2 + normal[1]**2)


def draw_variable_thickness_path(x, y, thickness, normals_distance=1e-1):
    sketch = get_empty_sketch()
    n = x.shape
    
    # if type(thickness) == float:
    #     thickness = np.full(n, thickness)
        
    assert y.shape == n
    # assert thickness.shape == n
    
    normal = get_unit_normal(x[0], y[0], x[1], y[1])
    normal_next = None
    rest = 0.0
    for i in range(x.shape[0] - 1):
        normal_next = get_unit_normal(x[i], y[i], x[i+1], y[i+1])
        # sketch.line(x[i], y[i], x[i] + 0.1*normal[0], y[i] + 0.1*normal[1])
        
        dx, dy = x[i+1] - x[i], y[i+1] - y[i]
        d = np.sqrt(dx**2 + dy**2)
        n_normals = 0
        # d < normals_distance -> 0
        # normals_distance < d < rest + normal_distance -> 1
        # rest + normal_distance < d < rest + 2*normal_distance -> 2
        if d > normals_distance:  # at least one
            n_normals = int(np.floor((d - rest) / normals_distance)) + 1
        for j in range(n_normals):
            # x_normal_center, y_normal_center = x[i] + dx * (rest + j / n_normals), y[i] + dy * (rest + j / n_normals)
            # dx * j / n_normals + rest_x, rest_x = dx / d * rest
            # h = rest, x = h * cos(theta), y = h * sin(theta), theta = arctan2(dy, dx)
            d_j = rest + normals_distance * j
            line_frac = d_j / d
            x_normal_center, y_normal_center = x[i] + dx / d * d_j, y[i] + dy / d * d_j
            normal_interp = normal + line_frac * (normal_next - normal)
            sketch.line(x_normal_center - thickness*normal_interp[0], y_normal_center - thickness*normal_interp[1], 
                        x_normal_center + thickness*normal_interp[0], y_normal_center + thickness*normal_interp[1])
        print(f"\n{i} d={d:.6f}, n_normals={n_normals}, rest={rest:.6f}")
        rest += n_normals * normals_distance - d
        # if n_normals > 0:
            # rest += np.fmod(d - rest, (n_normals - 1) * normals_distance)  # new rest
            # rest = d - rest - (n_normals - 1) * normals_distance
            # rest = normals_distance - d + (n_normals - 1) * normals_distance
            # rest = n_normals * normals_distance - d

        # elif n_normals == 1:
        #     rest = d - rest - normals_distance
        # else:
        #     rest += d
            # rest:
            # rest = d - rest - (n_normals - 1) * 
        print(d, (n_normals - 1) * normals_distance, rest)
        
        # sketch.line(x[i], y[i], x[i+1], y[i+1])
        sketch.line(x[i] + thickness*normal[0], y[i] + thickness*normal[1], x[i+1] + thickness*normal_next[0], y[i+1] + thickness*normal_next[1])
        sketch.line(x[i] - thickness*normal[0], y[i] - thickness*normal[1], x[i+1] - thickness*normal_next[0], y[i+1] - thickness*normal_next[1])
        
        normal = normal_next
    
    # Extra line on end segment:
    sketch.line(x[-1] - thickness*normal[0], y[-1] - thickness*normal[1], 
                x[-1] + thickness*normal[0], y[-1] + thickness*normal[1])
    return sketch
        

class SixfeetSketch(vsketch.SketchClass):
    # Sketch parameters:
    debug_show_shapes = vsketch.Param(False)
    occult = vsketch.Param(False)
    scale = vsketch.Param(1.0)
    
    def draw_debug_shapes(self, vsk):
        vsk.stroke(5)
        with vsk.pushMatrix():
            vsk.translate(0,-2)
            
            for x in range(20):
                for y in range(10):
                    vsk.circle(0.5 * x, 0.5 * y, radius=1e-2)
            
            vsk.sketch(draw_triangle(0, -1, width=0.5, height=0.5))     
            vsk.sketch(draw_triangle(1, -1.25, width=0.5, height=1.0))     
            vsk.sketch(draw_shaded_triangle(2, -1.25, width=0.5, height=1.0, fill_distance=0.1))     
            vsk.sketch(draw_filled_triangle(3, -1, width=0.5, height=0.5))     
                           
            vsk.sketch(draw_shaded_circle(0, 0, radius=0.375, fill_distance=0.1, angle=np.deg2rad(45)))
            vsk.sketch(draw_shaded_rect(1, 0, width=0.7, height=0.5, fill_distance=0.1, angle=np.deg2rad(45)))
            vsk.sketch(draw_filled_circle(2, 0, radius=0.375))
            vsk.sketch(draw_filled_rect(3, 0, 0.7, 0.5))
            vsk.sketch(draw_circle(4, 0, radius=0.375))
            vsk.sketch(draw_rect(5, 0, 0.7, 0.5))
            vsk.sketch(draw_pole(6, 0, 0.1, 1, 0.1))
            vsk.sketch(draw_flag(7, 0, 0.1, 1, 0.5, 0.3, right=False, triangular=True))
            vsk.sketch(draw_line(8, 0, 1))
            vsk.sketch(draw_thick_line(9, 0, 1))
            vsk.sketch(draw_filled_triangle(10, -1, width=6e-2, height=2))     
                        
            vsk.sketch(draw_dot_circle(0, 1, radius=0.375, radius_inner=0.15))
            vsk.sketch(draw_partial_filled_circle(1, 1, 0.375, fill_gain=0.5))
            vsk.sketch(draw_dot_shaded_rect(2, 1, 0.7, 0.6, dot_distance=6e-2))
            vsk.sketch(draw_dot_shaded_rect(3, 1, 0.7, 0.6, dot_distance=6e-2, vsk=vsk, noise_gain=0.4, noise_freq=4.0))
            vsk.sketch(draw_dot_shaded_circle(4, 1, 0.375, dot_distance=6e-2))
            vsk.sketch(draw_dot_shaded_circle(5, 1, 0.375, dot_distance=6e-2, vsk=vsk, noise_gain=0.4, noise_freq=4.0))
            
            vsk.sketch(draw_speckled_shaded_circle(8.5, 1, radius=0.75, density=200.0))
            vsk.sketch(draw_dash_shaded_circle(6.5, 1, radius=0.75, dash_length_gain=0.065, 
                                               padding_gain=0.03, N_tries=15000, N_allowed_fails=2000))
            
            vsk.sketch(draw_dash_shaded_rect(0, 3, width=1.0, height=2.5, dash_length_gain=0.08, 
                                             padding_gain=0.05, N_tries=2000, N_allowed_fails=1000))
            vsk.sketch(draw_speckled_shaded_rect(1.5, 3, width=1.0, height=2.5, density=200.0))

            
            vsk.sketch(draw_dot_evenly_shaded_rect(3, 3, width=1.0, height=2.5, density=600, dot_radius=0.03))
            vsk.sketch(draw_dot_evenly_shaded_rect(4.5, 2.25, width=1.0, height=1.0, density=200, dot_radius=0.05))
            vsk.sketch(draw_dash_shaded_triangle(4.5, 3.5, width=1.0, height=1.0, dash_length_gain=0.065, 
                        padding_gain=0.05, N_tries=2000, N_allowed_fails=1000))
            vsk.sketch(draw_dot_evenly_shaded_triangle(6, 3.25, width=1.0, height=1.5, density=300, dot_radius=0.04))
            vsk.sketch(draw_dot_evenly_shaded_circle(8.5, 3, radius=0.75, density=600, dot_radius=0.03))
            
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale)
        
        if self.debug_show_shapes:
            self.draw_debug_shapes(vsk)
        
        
        # rect = draw_dash_shaded_rect(0, 0, width=2.0, height=4.0, dash_length_gain=0.08, 
        #                             padding_gain=0.05, N_tries=2000, N_allowed_fails=1000)
        # rotate_and_draw_sketch(vsk, rect, 2, 3, np.deg2rad(2))
        
        # line = draw_thick_line(0, 0, 3, width=5e-2)
        # rotate_and_draw_sketch(vsk, line, 0, 0, np.deg2rad(145))
        
        # line = draw_line(0, 0, 2)
        # rotate_and_draw_sketch(vsk, line, 0, 0, np.deg2rad(-35))
        
        # vsk.sketch(draw_partial_filled_circle(0, 0, 1, fill_gain=0.45))
        
        # self.draw_test_1(vsk)
        self.test_variable_thickness_line(vsk)

        if self.occult:
            vsk.vpype("occult -i")
    
    
    def draw_test_1(self, vsk):
        line = draw_thick_line(0, 0, 4, width=0.1)
        rotate_and_draw_sketch(vsk, line, 2, 0, np.deg2rad(-35))
        
        line = draw_line(0, 0, 6)
        rotate_and_draw_sketch(vsk, line, 0, 0, np.deg2rad(22))
        
        rect = draw_dot_evenly_shaded_rect(0, 0, width=1.0, height=3.0, density=200, dot_radius=0.05)
        rotate_and_draw_sketch(vsk, rect, -3, -3, np.deg2rad(-60))
        
        tri = draw_filled_triangle(0, 0, width=6e-2, height=5)
        rotate_and_draw_sketch(vsk, tri, -3, -6, np.deg2rad(-80))
        
        line = draw_line(0, 0, 1)
        rotate_and_draw_sketch(vsk, line, -5, -6, np.deg2rad(25))
        line = draw_line(0, 0, 1.2)
        rotate_and_draw_sketch(vsk, line, -4.9, -5.9, np.deg2rad(25))
        line = draw_line(0, 0, 1)
        rotate_and_draw_sketch(vsk, line, -4.7, -5.9, np.deg2rad(25))
        
        circle = draw_circle(1.5, -4, radius=0.375)
        vsk.sketch(circle)
        
        circle = draw_circle(-1, -3, radius=2.8)
        vsk.sketch(circle)
        
        rect = draw_filled_rect(0, 0, 0.5, 1.2)
        rotate_and_draw_sketch(vsk, rect, -2.4, -3.4, np.deg2rad(11))

    def test_variable_thickness_line(self, vsk):
        n = 100
        t = np.linspace(0, 8 * np.pi, n, endpoint=True)
        x, y = 1.0 * np.sqrt(t) * (np.cos(t), np.sin(t))
        thickness = 0.2
        vsk.sketch(draw_variable_thickness_path(x, y, thickness, normals_distance=0.1))
    
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    SixfeetSketch.display()
