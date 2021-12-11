import vsketch
from matplotlib import pyplot as plt
import numpy as np
import os
from map_helpers import draw_border, draw_map, draw_title, draw_water, interp_map, get_map_scale_factor, show_map_image, trunc_map_lower, smoothen_map, crop_map

class MapSketch(vsketch.SketchClass):

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        # vsk.detail("1mm")
        vsk.penWidth("2mm")
        
        skips = 1
        padding = 1.75
        levels = 35
        sigma = 1.0
        interp_scale = 2
        water_level = -0.4
        
        path = "map/fyn/"
        file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        all_data = np.zeros((4, 1000, 1000))
        for i, file_name in enumerate(file_names):
            img = np.loadtxt(path + file_name, skiprows=6)
            all_data[i] = img
            
        img = np.zeros((2000, 2000))
        img[:1000, :1000] = all_data[1]
        img[1000:, :1000] = all_data[0]
        img[:1000, 1000:] = all_data[3]
        img[1000:, 1000:] = all_data[2]
            
        x_lim = [960, 1540]
        y_0 = 450
        y_1 = int(y_0 + (x_lim[1] - x_lim[0]) / (21 - 2*padding) * (29.7 - 2*padding))
        img = crop_map(img, x_lim, [y_0, y_1])

        img = img[::skips,::skips]
        
        # img = trunc_map_lower(img, water_level)
        
        # img = interp_map(img, interp_scale)

        img = smoothen_map(img, sigma)
        
        img = trunc_map_lower(img, water_level)
        
        show_map_image(img)
        
        draw_border(vsk, padding=padding)
                
        scale = get_map_scale_factor(vsk, padding, img.shape[1])

        draw_map(img, vsk, levels=levels, scale=scale, offset=[padding, padding])

        period = int(img.shape[1] / 35)
        draw_water(vsk, img, period, water_level=1e-1, scale=scale, offset=[padding, padding], radius=0.04)

        draw_title(vsk, "FYNS HOVED", padding=padding)
        
        vsk.vpype("linemerge linesimplify reloop linesort")
        vsk.save("map/output/map_fyn_2.svg")
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    MapSketch.display()
