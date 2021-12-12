import vsketch
import numpy as np
import iso

# Ideas:
# - isometric grid with towers, bridges, slopes etc.
# - 3d tubes using circles


class RandomIsoSketch(vsketch.SketchClass):
    
    draw_axes = vsketch.Param(False)
    draw_grid = vsketch.Param(False)
    draw_debug = vsketch.Param(False)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        # DO NO OCCULT THINGS HERE
        
        if self.draw_axes: iso.draw_axes(vsk, x_axis_length=7, y_axis_length=7)
        if self.draw_grid: iso.draw_grid(vsk, x_size=6, y_size=6)
        
        # shape_0 = isoShape(2, 1, 0, 2, 1, 2)
        # shape_1 = isoShape(1, 2, 0, 3, 1, 1)
        # shape_2 = isoShape(1, 4, 0, 1, 1, 1)
        # shape_3 = isoShape(2, 3, 0, 1, 2, 1)
        # shapes = [shape_0, shape_1, shape_2, shape_3]
        # random.shuffle(shapes)
        
        
        N = 50
        draw_order = None
        while draw_order is None:  # this is really bad, but a quick way to avoid the occasional cycle
            try:
                shapes = []
                for i in range(N):
                    x = np.random.uniform(0, 10)
                    y = np.random.uniform(0, 10)
                    z = np.random.uniform(0, 19)
                    x_size = np.random.uniform(0.25, 3.0)
                    y_size = np.random.uniform(0.25, 3.0)
                    z_size = np.random.uniform(0.25, 3.0)
                    shapes.append(iso.isoShape(x, y, z, x_size, y_size, z_size))
        
                draw_order = iso.get_draw_order(shapes)
            except:
                pass
        
        for i in draw_order:
            shapes[i].draw(vsk, dx_shade=0.15, dy_shade=0.075)
            if self.draw_debug: shapes[i].draw_debug(vsk, offset=0.2*i)
            
        vsk.vpype("occult -i")
        
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    RandomIsoSketch.display()
