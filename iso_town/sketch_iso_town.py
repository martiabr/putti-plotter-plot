import vsketch
import numpy as np
import iso

# Idea:
# - Use perlin noise to generate terrain of some sort, maybe islands and/or mountains or something
# - Generate structures on terrain - randomly or using some algorithm like a variation of wave function collapse. Maybe some mystical monoliths or something, stonehenge. Or towers, bridges, houses, ...

# start by generating simple terrain and figure out how to join textures? And add additional shapes to draw structues

# - We have code for finding the intersection plane. We need code for checking if shapes are touching (and not simple overlapping).
# - With that we can create a graph of all shapes that are touching. We can 1. add type id to shapes and 2. enable/disable connecting textures
#   in order to have control over what is connected or not.
# - Given two shapes that are touching and which side is touching, we need to calculate how much to "remove" of the outline on that side.
#   Uncertain if we somehow still can get occult to work...

# The simpler solution for now is just to add shading on top as well...

class IsoTownSketch(vsketch.SketchClass):
    
    draw_axes = vsketch.Param(False)
    draw_grid = vsketch.Param(False)
    draw_debug = vsketch.Param(False)
    draw_shading = vsketch.Param(False)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        # vsk.noiseDetail()
        
        if self.draw_axes: iso.draw_axes(vsk, x_axis_length=7, y_axis_length=7)
        if self.draw_grid: iso.draw_grid(vsk, x_size=6, y_size=6)
        
        vsk.vpype("occult -i")
        
        
        shapes = []
        shapes.append(iso.isoShape(1, 1, 0, 1, 1, 1))
        shapes.append(iso.isoShape(0, 2, 0, 3, 1, 1))
        
        # shapes = []
        # amplitude = 7
        # gain = 0.2
        # z_grid = (amplitude * vsk.noise(gain * np.linspace(0, 9, 10), gain * np.linspace(0, 9, 10))).astype(int)
        # for x, row in enumerate(z_grid):
        #     for y, z_top in enumerate(row):
        #         for z in range(z_top):
        #             shapes.append(iso.isoShape(x, y, z, 1, 1, 1))
        
        if self.draw_shading:
            dx_shade=0.15
            dy_shade=0.075
            dz_shade=1.0 + 1e-6
        else:
            dx_shade = dy_shade = dz_shade = None
        
        draw_order = iso.get_draw_order(shapes)
        for i in draw_order:
            shapes[i].draw(vsk, dx_shade=dx_shade, dy_shade=dy_shade, dz_shade=dz_shade)
            if self.draw_debug: shapes[i].draw_debug(vsk, offset=0.2*i)
        
        
        vsk.vpype("occult -i")
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    IsoTownSketch.display()
