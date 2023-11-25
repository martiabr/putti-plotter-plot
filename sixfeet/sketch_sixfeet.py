import vsketch
import numpy as np
from numpy.random import default_rng
import random
import winsound
from plotter_shapes.plotter_shapes import *
from plotter_shapes.variable_width_path import draw_variable_width_path  


class SixfeetSketch(vsketch.SketchClass):
    # Sketch parameters:
    debug_show_shapes = vsketch.Param(False)
    debug_draw = vsketch.Param(False)
    occult = vsketch.Param(False)
    scale = vsketch.Param(1.0)
    
    rng = default_rng()
    WIDTH_FULL = 21
    HEIGHT_FULL = 29.7
    
    area_type = vsketch.Param("RECT", choices=["RECT", "CIRCLE", "BLOB"])
    width = vsketch.Param(6.0)
    height = vsketch.Param(6.0)

    # Circle empty:
    num_circ_min = vsketch.Param(0, min_value=0)
    num_circ_max = vsketch.Param(5, min_value=0)
    circ_rad_min = vsketch.Param(0.05, min_value=0)
    circ_rad_max = vsketch.Param(0.5, min_value=0)
    
    # Circle shaded:
    num_circ_shaded_min = vsketch.Param(0, min_value=0)
    num_circ_shaded_max = vsketch.Param(5, min_value=0)
    circ_shaded_rad_min = vsketch.Param(0.05, min_value=0)
    circ_shaded_rad_max = vsketch.Param(0.5, min_value=0)
    circ_shaded_fill_dist_min = vsketch.Param(0.05, min_value=0)
    circ_shaded_fill_dist_max = vsketch.Param(0.2, min_value=0)
        
    # Circle fill:
    num_circ_fill_min = vsketch.Param(0, min_value=0)
    num_circ_fill_max = vsketch.Param(5, min_value=0)
    circ_fill_rad_min = vsketch.Param(0.05, min_value=0)
    circ_fill_rad_max = vsketch.Param(0.5, min_value=0)
    
    # Circle dash filled:
    num_circ_dash_min = vsketch.Param(0, min_value=0)
    num_circ_dash_max = vsketch.Param(5, min_value=0)
    circ_dash_rad_min = vsketch.Param(0.05, min_value=0)
    circ_dash_rad_max = vsketch.Param(0.5, min_value=0)
    
    # Circle dot filled:
    num_circ_dot_min = vsketch.Param(0, min_value=0)
    num_circ_dot_max = vsketch.Param(5, min_value=0)
    circ_dot_rad_min = vsketch.Param(0.05, min_value=0)
    circ_dot_rad_max = vsketch.Param(0.5, min_value=0)
    circ_dot_dens_min = vsketch.Param(400.0, min_value=0)
    circ_dot_dens_max = vsketch.Param(800.0, min_value=0)
    circ_dot_dot_rad_min = vsketch.Param(0.02, min_value=0)
    circ_dot_dot_rad_max = vsketch.Param(0.06, min_value=0)
    
    # Circle half filled:
    num_circ_half_fill_min = vsketch.Param(0, min_value=0)
    num_circ_half_fill_max = vsketch.Param(5, min_value=0)
    circ_half_fill_rad_min = vsketch.Param(0.05, min_value=0)
    circ_half_fill_rad_max = vsketch.Param(0.5, min_value=0)
    circ_half_fill_gain_min = vsketch.Param(0.4, min_value=0)
    circ_half_fill_gain_max = vsketch.Param(0.6, min_value=0)
    
    # Circle outer filled:
    num_circ_outer_fill_min = vsketch.Param(0, min_value=0)
    num_circ_outer_fill_max = vsketch.Param(5, min_value=0)
    circ_outer_fill_rad_min = vsketch.Param(0.05, min_value=0)
    circ_outer_fill_rad_max = vsketch.Param(0.5, min_value=0)
    circ_outer_fill_gain_min = vsketch.Param(0.0, min_value=0)
    circ_outer_fill_gain_max = vsketch.Param(0.5, min_value=0)
    
    # Line:
    num_line_min = vsketch.Param(0, min_value=0)
    num_line_max = vsketch.Param(5, min_value=0)
    line_length_min = vsketch.Param(1.0, min_value=0)
    line_length_max = vsketch.Param(5.0, min_value=0)
    
    # Thick line:
    num_thick_line_min = vsketch.Param(0, min_value=0)
    num_thick_line_max = vsketch.Param(5, min_value=0)
    thick_line_length_min = vsketch.Param(1.0, min_value=0)
    thick_line_length_max = vsketch.Param(5.0, min_value=0)
    
    # Triangle line:
    num_tri_line_min = vsketch.Param(0, min_value=0)
    num_tri_line_max = vsketch.Param(5, min_value=0)
    tri_line_length_min = vsketch.Param(1.0, min_value=0)
    tri_line_length_max = vsketch.Param(5.0, min_value=0)
    tri_line_width_gain_min = vsketch.Param(0.01, min_value=0)
    tri_line_width_gain_max = vsketch.Param(0.02, min_value=0)
    
    # Rect empty:
    num_rect_min = vsketch.Param(0, min_value=0)
    num_rect_max = vsketch.Param(5, min_value=0)
    rect_width_min = vsketch.Param(0.1, min_value=0)
    rect_width_max = vsketch.Param(1.0, min_value=0)
    rect_height_gain_min = vsketch.Param(0.5, min_value=0)
    rect_height_gain_max = vsketch.Param(2.0, min_value=0)
    
    # Rect shaded:
    num_rect_shaded_min = vsketch.Param(0, min_value=0)
    num_rect_shaded_max = vsketch.Param(5, min_value=0)
    rect_shaded_width_min = vsketch.Param(0.1, min_value=0)
    rect_shaded_width_max = vsketch.Param(1.0, min_value=0)
    rect_shaded_height_gain_min = vsketch.Param(0.5, min_value=0)
    rect_shaded_height_gain_max = vsketch.Param(2.0, min_value=0)
    rect_shaded_fill_dist_min = vsketch.Param(0.05, min_value=0)
    rect_shaded_fill_dist_max = vsketch.Param(0.2, min_value=0)
    
    # Rect filled:
    num_rect_fill_min = vsketch.Param(0, min_value=0)
    num_rect_fill_max = vsketch.Param(5, min_value=0)
    rect_fill_width_min = vsketch.Param(0.1, min_value=0)
    rect_fill_width_max = vsketch.Param(1.0, min_value=0)
    rect_fill_height_gain_min = vsketch.Param(0.5, min_value=0)
    rect_fill_height_gain_max = vsketch.Param(2.0, min_value=0)
    
    # Rect dash filled:
    num_rect_dash_min = vsketch.Param(0, min_value=0)
    num_rect_dash_max = vsketch.Param(5, min_value=0)
    rect_dash_width_min = vsketch.Param(0.1, min_value=0)
    rect_dash_width_max = vsketch.Param(1.0, min_value=0)
    rect_dash_height_gain_min = vsketch.Param(0.5, min_value=0)
    rect_dash_height_gain_max = vsketch.Param(2.0, min_value=0)
    
    # Rect dot filled:
    num_rect_dot_min = vsketch.Param(0, min_value=0)
    num_rect_dot_max = vsketch.Param(5, min_value=0)
    rect_dot_width_min = vsketch.Param(0.1, min_value=0)
    rect_dot_width_max = vsketch.Param(1.0, min_value=0)
    rect_dot_height_gain_min = vsketch.Param(0.5, min_value=0)
    rect_dot_height_gain_max = vsketch.Param(2.0, min_value=0)
    rect_dot_dens_min = vsketch.Param(400.0, min_value=0)
    rect_dot_dens_max = vsketch.Param(800.0, min_value=0)
    rect_dot_rad_min = vsketch.Param(0.02, min_value=0)
    rect_dot_rad_max = vsketch.Param(0.06, min_value=0)
    
    # Triangle empty:
    num_tri_min = vsketch.Param(0, min_value=0)
    num_tri_max = vsketch.Param(5, min_value=0)
    tri_width_min = vsketch.Param(0.1, min_value=0)
    tri_width_max = vsketch.Param(1.0, min_value=0)
    tri_height_gain_min = vsketch.Param(0.5, min_value=0)
    tri_height_gain_max = vsketch.Param(2.0, min_value=0)
    
    # Triangle shaded:
    num_tri_shaded_min = vsketch.Param(0, min_value=0)
    num_tri_shaded_max = vsketch.Param(5, min_value=0)
    tri_shaded_width_min = vsketch.Param(0.1, min_value=0)
    tri_shaded_width_max = vsketch.Param(1.0, min_value=0)
    tri_shaded_height_gain_min = vsketch.Param(0.5, min_value=0)
    tri_shaded_height_gain_max = vsketch.Param(2.0, min_value=0)
    tri_shaded_fill_dist_min = vsketch.Param(0.05, min_value=0)
    tri_shaded_fill_dist_max = vsketch.Param(0.2, min_value=0)
    
    # Triangle filled:
    num_tri_fill_min = vsketch.Param(0, min_value=0)
    num_tri_fill_max = vsketch.Param(5, min_value=0)
    tri_fill_width_min = vsketch.Param(0.1, min_value=0)
    tri_fill_width_max = vsketch.Param(1.0, min_value=0)
    tri_fill_height_gain_min = vsketch.Param(0.5, min_value=0)
    tri_fill_height_gain_max = vsketch.Param(2.0, min_value=0)
    
    # Triangle dash filled:
    num_tri_dash_min = vsketch.Param(0, min_value=0)
    num_tri_dash_max = vsketch.Param(5, min_value=0)
    tri_dash_width_min = vsketch.Param(0.1, min_value=0)
    tri_dash_width_max = vsketch.Param(1.0, min_value=0)
    tri_dash_height_gain_min = vsketch.Param(0.5, min_value=0)
    tri_dash_height_gain_max = vsketch.Param(2.0, min_value=0)
    
    # Triangle dot filled:
    num_tri_dot_min = vsketch.Param(0, min_value=0)
    num_tri_dot_max = vsketch.Param(5, min_value=0)
    tri_dot_width_min = vsketch.Param(0.1, min_value=0)
    tri_dot_width_max = vsketch.Param(1.0, min_value=0)
    tri_dot_height_gain_min = vsketch.Param(0.5, min_value=0)
    tri_dot_height_gain_max = vsketch.Param(2.0, min_value=0)
    tri_dot_dens_min = vsketch.Param(400.0, min_value=0)
    tri_dot_dens_max = vsketch.Param(800.0, min_value=0)
    tri_dot_rad_min = vsketch.Param(0.02, min_value=0)
    tri_dot_rad_max = vsketch.Param(0.06, min_value=0)
    
    
    def draw_shape_composition(self, vsk, shapes):
        for shape in shapes:
            # TODO: handle array of shapes in array
            with vsk.pushMatrix():
                x, y, theta = self.rng.uniform([-0.5 * self.width, -0.5 * self.height, -np.pi],
                                               [0.5 * self.width, 0.5 * self.height, np.pi])
                rotate_and_draw_sketch(vsk, shape, x, y, theta)
    
    def draw_debug_shapes(self, vsk):
        vsk.stroke(5)
        with vsk.pushMatrix():
            vsk.translate(0,-2)
            
            for x in range(21):
                for y in range(-3, 10):
                    vsk.circle(0.5 * x, 0.5 * y, radius=1e-2)
            
            vsk.sketch(draw_triangle(0, -1, width=0.5, height=0.5))     
            vsk.sketch(draw_triangle(1, -1.25, width=0.5, height=1.0))     
            vsk.sketch(draw_shaded_triangle(2, -1.25, width=0.5, height=1.0, fill_distance=0.1))     
            vsk.sketch(draw_filled_triangle(3, -1, width=0.5, height=0.5))
            vsk.sketch(draw_thick_circle(4, -1, radius=0.375, fill_gain=0.25))
                           
            vsk.sketch(draw_shaded_circle(0, 0, radius=0.375, fill_distance=0.1, angle=np.deg2rad(45)))
            vsk.sketch(draw_shaded_rect(1, 0, width=0.7, height=0.5, fill_distance=0.1, angle=np.deg2rad(45)))
            vsk.sketch(draw_filled_circle(2, 0, radius=0.375))
            vsk.sketch(draw_filled_rect(3, 0, 0.7, 0.5))
            vsk.sketch(draw_circle(4, 0, radius=0.375))
            vsk.sketch(draw_rect(5, 0, 0.7, 0.5))
            vsk.sketch(draw_pole(6, 0, 0.1, 1, 0.1))
            vsk.sketch(draw_flag(7, 0, 0.1, 1, 0.5, 0.3, right=False, triangular=True))
            vsk.sketch(draw_line(8, -0.5, 1))
            vsk.sketch(draw_thick_line(9, -0.5, 1))
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
            
        ###
        
        shapes = []
        
        num_circ = self.rng.integers(self.num_circ_min, self.num_circ_max + 1)
        radii = self.rng.uniform(self.circ_rad_min, self.circ_rad_max, size=num_circ)
        for radius in radii:
            shapes.append(draw_circle(0, 0, radius=radius))
            
        num_circ_shaded = self.rng.integers(self.num_circ_shaded_min, self.num_circ_shaded_max + 1)
        radii = self.rng.uniform(self.circ_shaded_rad_min, self.circ_shaded_rad_max, size=num_circ_shaded)
        fill_dists = self.rng.uniform(self.circ_shaded_fill_dist_min, self.circ_shaded_fill_dist_max, size=num_circ_shaded)
        for radius, dist in zip(radii, fill_dists):
            shapes.append(draw_shaded_circle(0, 0, radius=radius, fill_distance=dist))
            
        num_circ_fill = self.rng.integers(self.num_circ_fill_min, self.num_circ_fill_max + 1)
        radii = self.rng.uniform(self.circ_fill_rad_min, self.circ_fill_rad_max, size=num_circ_fill)
        for radius in radii:
            shapes.append(draw_filled_circle(0, 0, radius=radius))
            
        num_circ_half_fill = self.rng.integers(self.num_circ_half_fill_min, self.num_circ_half_fill_max + 1)
        radii = self.rng.uniform(self.circ_half_fill_rad_min, self.circ_half_fill_rad_max, size=num_circ_half_fill)
        fills = self.rng.uniform(self.circ_half_fill_gain_min, self.circ_half_fill_gain_max, size=num_circ_half_fill)
        for radius, fill in zip(radii, fills):
            shapes.append(draw_partial_filled_circle(0, 0, radius=radius, fill_gain=fill))
            
        num_circ_outer_fill = self.rng.integers(self.num_circ_outer_fill_min, self.num_circ_outer_fill_max + 1)
        radii = self.rng.uniform(self.circ_outer_fill_rad_min, self.circ_outer_fill_rad_max, size=num_circ_outer_fill)
        fills = self.rng.uniform(self.circ_outer_fill_gain_min, self.circ_outer_fill_gain_max, size=num_circ_outer_fill)
        for radius, fill in zip(radii, fills):
            shapes.append(draw_thick_circle(0, 0, radius=radius, fill_gain=fill))
            
        num_circ_dot_fill = self.rng.integers(self.num_circ_dot_min, self.num_circ_dot_max + 1)
        radii = self.rng.uniform(self.circ_dot_rad_min, self.circ_dot_rad_max, size=num_circ_dot_fill)
        dot_radii = self.rng.uniform(self.circ_dot_dot_rad_min, self.circ_dot_dot_rad_max, size=num_circ_dot_fill)
        densities = self.rng.uniform(self.circ_dot_dens_min, self.circ_dot_dens_max, size=num_circ_dot_fill)
        for rad, dot_rad, density in zip(radii, dot_radii, densities):
            shapes.append(draw_dot_evenly_shaded_circle(0, 0, radius=rad, density=density, dot_radius=dot_rad))
            
        num_lines = self.rng.integers(self.num_line_min, self.num_line_max + 1)
        lengths = self.rng.uniform(self.line_length_min, self.line_length_max, size=num_lines)
        for length in lengths:
            shapes.append(draw_line(0, 0, length=length))
            
        num_thick_lines = self.rng.integers(self.num_thick_line_min, self.num_thick_line_max + 1)
        lengths = self.rng.uniform(self.thick_line_length_min, self.thick_line_length_max, size=num_thick_lines)
        for length in lengths:
            shapes.append(draw_thick_line(0, 0, length=length))
            
        num_tri_lines = self.rng.integers(self.num_tri_line_min, self.num_tri_line_max + 1)
        lengths = self.rng.uniform(self.tri_line_length_min, self.tri_line_length_max, size=num_tri_lines)
        widths = self.rng.uniform(self.tri_line_width_gain_min, self.tri_line_width_gain_max, size=num_tri_lines) * lengths
        for length, width in zip(lengths, widths):
            shapes.append(draw_triangle_line(0, 0, length=length, width=width))

        num_rect = self.rng.integers(self.num_rect_min, self.num_rect_max + 1)
        widths = self.rng.uniform(self.rect_width_min, self.rect_width_max, size=num_rect)
        heights = self.rng.uniform(self.rect_height_gain_min, self.rect_height_gain_max, size=num_rect) * widths
        for width, height in zip(widths, heights):
            shapes.append(draw_rect(0, 0, width=width, height=height))
            
        num_rect_shaded = self.rng.integers(self.num_rect_shaded_min, self.num_rect_shaded_max + 1)
        widths = self.rng.uniform(self.rect_shaded_width_min, self.rect_shaded_width_max, size=num_rect_shaded)
        heights = self.rng.uniform(self.rect_shaded_height_gain_min, self.rect_shaded_height_gain_max, size=num_rect_shaded) * widths
        fill_dists = self.rng.uniform(self.rect_shaded_fill_dist_min, self.rect_shaded_fill_dist_max, size=num_rect_shaded)
        for width, height, dist in zip(widths, heights, fill_dists):
            shapes.append(draw_shaded_rect(0, 0, width=width, height=height, fill_distance=dist))
            
        num_rect_fill = self.rng.integers(self.num_rect_fill_min, self.num_rect_fill_max + 1)
        widths = self.rng.uniform(self.rect_fill_width_min, self.rect_fill_width_max, size=num_rect_fill)
        heights = self.rng.uniform(self.rect_fill_height_gain_min, self.rect_fill_height_gain_max, size=num_rect_fill) * widths
        for width, height in zip(widths, heights):
            shapes.append(draw_filled_rect(0, 0, width=width, height=height))
        
        num_rect_dot_fill = self.rng.integers(self.num_rect_dot_min, self.num_rect_dot_max + 1)
        widths = self.rng.uniform(self.rect_dot_width_min, self.rect_dot_width_max, size=num_rect_dot_fill)
        heights = self.rng.uniform(self.rect_dot_height_gain_min, self.rect_dot_height_gain_max, size=num_rect_dot_fill) * widths
        dot_radii = self.rng.uniform(self.rect_dot_rad_min, self.rect_dot_rad_max, size=num_rect_dot_fill)
        densities = self.rng.uniform(self.rect_dot_dens_min, self.rect_dot_dens_max, size=num_rect_dot_fill)
        for width, height, rad, dens in zip(widths, heights, dot_radii, densities):
            shapes.append(draw_dot_evenly_shaded_rect(0, 0,  width=width, height=height, density=dens, dot_radius=rad))
        
        # num_rect_dash_fill = self.rng.integers(self.num_rect_dash_min, self.num_rect_dash_max + 1)
        # widths = self.rng.uniform(self.rect_dash_width_min, self.rect_dash_width_max, size=num_rect_dash_fill)
        # heights = self.rng.uniform(self.rect_dash_height_gain_min, self.rect_dash_height_gain_max, size=num_rect_dash_fill) * widths
        # for width, height in zip(widths, heights):
        #     shapes.append(draw_dash_shaded_rect(0, 0,  width=width, height=height))
        
        num_tri = self.rng.integers(self.num_tri_min, self.num_tri_max + 1)
        widths = self.rng.uniform(self.tri_width_min, self.tri_width_max, size=num_tri)
        heights = self.rng.uniform(self.tri_height_gain_min, self.tri_height_gain_max, size=num_tri) * widths
        for width, height in zip(widths, heights):
            shapes.append(draw_triangle(0, 0, width=width, height=height))
            
        num_tri_shaded = self.rng.integers(self.num_tri_shaded_min, self.num_tri_shaded_max + 1)
        widths = self.rng.uniform(self.tri_shaded_width_min, self.tri_shaded_width_max, size=num_tri_shaded)
        heights = self.rng.uniform(self.tri_shaded_height_gain_min, self.tri_shaded_height_gain_max, size=num_tri_shaded) * widths
        fill_dists = self.rng.uniform(self.tri_shaded_fill_dist_min, self.tri_shaded_fill_dist_max, size=num_tri_shaded)
        for width, height, dist in zip(widths, heights, fill_dists):
            shapes.append(draw_shaded_triangle(0, 0, width=width, height=height, fill_distance=dist))
           
        num_tri_fill = self.rng.integers(self.num_tri_fill_min, self.num_tri_fill_max + 1)
        widths = self.rng.uniform(self.tri_fill_width_min, self.tri_fill_width_max, size=num_tri_fill)
        heights = self.rng.uniform(self.tri_fill_height_gain_min, self.tri_fill_height_gain_max, size=num_tri_fill) * widths
        for width, height in zip(widths, heights):
            shapes.append(draw_filled_triangle(0, 0, width=width, height=height))   
        
        num_tri_dot_fill = self.rng.integers(self.num_tri_dot_min, self.num_tri_dot_max + 1)
        widths = self.rng.uniform(self.tri_dot_width_min, self.tri_dot_width_max, size=num_tri_dot_fill)
        heights = self.rng.uniform(self.tri_dot_height_gain_min, self.tri_dot_height_gain_max, size=num_tri_dot_fill) * widths
        dot_radii = self.rng.uniform(self.tri_dot_rad_min, self.tri_dot_rad_max, size=num_tri_dot_fill)
        densities = self.rng.uniform(self.tri_dot_dens_min, self.tri_dot_dens_max, size=num_tri_dot_fill)
        for width, height, rad, dens in zip(widths, heights, dot_radii, densities):
            shapes.append(draw_dot_evenly_shaded_triangle(0, 0,  width=width, height=height, density=dens, dot_radius=rad))
           
           
        if self.occult:
            random.shuffle(shapes)

            
        self.draw_shape_composition(vsk, shapes)
        
        ###
        
        # self.draw_test_1(vsk)
        # self.draw_test_2(vsk)
        # self.test_variable_thickness_line(vsk)
        
        if self.debug_draw:
            self.draw_debug_info(vsk)

        if self.occult:
            vsk.vpype("occult -i")
            
        winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS) 
    
    def draw_test_2(self, vsk):
        shapes = []
        
        circle = draw_circle(0, 0, radius=2.8)
        shapes.append(circle)
        
        line = draw_thick_line(0, 0, 4, width=0.1)
        shapes.append(line)
        
        line = draw_line(0, 0, 6)
        shapes.append(line)
        
        rect = draw_dot_evenly_shaded_rect(0, 0, width=1.0, height=3.0, density=200, dot_radius=0.05)
        shapes.append(rect)
        
        shape = get_empty_sketch()
        tri = draw_filled_triangle(0, 0, width=6e-2, height=5)
        shape.sketch(tri)
        rotate_and_draw_sketch(shape, draw_line(0, 0, 1), x=0, y=-1.6, angle=0.47*np.pi)
        rotate_and_draw_sketch(shape, draw_line(0, 0, 1.2), x=0, y=-1.5, angle=0.47*np.pi)
        rotate_and_draw_sketch(shape, draw_line(0, 0, 1), x=0.1, y=-1.4, angle=0.47*np.pi)
        shapes.append(shape)
        
        circle = draw_circle(0, 0, radius=0.375)
        shapes.append(circle)
        
        rect = draw_filled_rect(0, 0, 0.5, 1.2)
        shapes.append(rect)
        
        circle = draw_filled_circle(0, 0, radius=0.08)
        shapes.append(circle)
        
        self.draw_shape_composition(vsk, shapes)
        
    def draw_test_1(self, vsk):
        circle = draw_circle(-1, -3, radius=2.8)
        vsk.sketch(circle)
        
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
        
        rect = draw_filled_rect(0, 0, 0.5, 1.2)
        rotate_and_draw_sketch(vsk, rect, -2.4, -3.4, np.deg2rad(11))
        
        vsk.sketch(draw_filled_circle(-0.2, -1.5, radius=0.08))

    def test_variable_thickness_line(self, vsk):
        n = 800
        t = np.linspace(0, 8 * np.pi, n, endpoint=True)
        x, y = 0.7 * np.sqrt(t) * (np.cos(t), np.sin(t))
        width = 0.1 + 0.045 * np.sin(2.5 * t)
        vsk.sketch(draw_variable_width_path(x, y, width, normals_distance=0.02))
        
        width = 0.12 + 0.15 * (vsk.noise(1.0*t + 1000) - 0.5) + 0.04 * (vsk.noise(10.0*t) - 0.5)  # this doesnt work completely right. Need to resample to even distance to get noise correct
        for i in range(5):
            width[i] *= i / 5
            width[-i] *= i / 5 
        vsk.sketch(draw_variable_width_path(x, y + 8, width, normals_distance=0.02))
    
    def draw_debug_info(self, vsk):
        vsk.stroke(2)
        vsk.line(-0.5 * self.width, -0.5 * self.height, 0.5 * self.width, -0.5 * self.height)
        vsk.line(-0.5 * self.width, 0.5 * self.height, 0.5 * self.width, 0.5 * self.height)
        vsk.line(-0.5 * self.width, -0.5 * self.height, -0.5 * self.width, 0.5 * self.height)
        vsk.line(0.5 * self.width, -0.5 * self.height, 0.5 * self.width, 0.5 * self.height)
        vsk.stroke(1)
    
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    SixfeetSketch.display()
