import vsketch
import numpy as np
from plotter_shapes.plotter_shapes import *


class SixfeetSketch(vsketch.SketchClass):
    # Sketch parameters:
    debug_show_shapes = vsketch.Param(True)
    occult = vsketch.Param(False)
    scale = vsketch.Param(1.0)
    
    def draw_debug_shapes(self, vsk):
        vsk.stroke(5)
        with vsk.pushMatrix():
            vsk.translate(0,-2)
            
            for x in range(20):
                for y in range(10):
                    vsk.circle(0.5 * x, 0.5 * y, radius=1e-2)
                        
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
        
            
        if self.occult:
            vsk.vpype("occult -i")
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    SixfeetSketch.display()
