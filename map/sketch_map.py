import vsketch
from matplotlib import pyplot as plt
import numpy as np
import rasterio as rio
from map_helpers import draw_border, draw_map, draw_title, draw_water, interp_map, get_map_scale_factor, show_map_image, trunc_map_lower, smoothen_map, crop_map

class MapSketch(vsketch.SketchClass):

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        # vsk.detail("1mm")
        vsk.penWidth("2mm")
        
        skips = 5
        padding = 1.25
        levels = 50
        sigma = 0.1
        interp_scale = 2
        water_level = 1e-4
        
        img = rio.open('map/valle/dtm10/data/dtm10_6501_2_10m_z33.tif').read(1)
        
        x_lim = [3600, 4000]
        y_0 = 285
        y_1 = int(y_0 + (x_lim[1] - x_lim[0]) / (21 - 2*padding) * (29.7 - 2*padding))
        img = crop_map(img, x_lim, [y_0, y_1])
        
        img = img[::skips,::skips]
        
        # img = interp_map(img, interp_scale)
        
        img = smoothen_map(img, sigma)
        
        img = trunc_map_lower(img, water_level)
        
        # draw_border(vsk, padding=padding)
                
        scale = get_map_scale_factor(vsk, padding, img.shape[1])
        
        draw_map(img, vsk, levels=levels, scale=scale, offset=[padding, padding])
        
        period = int(img.shape[1] / 60)
        draw_water(vsk, img, period, water_level=1e-1, scale=scale, offset=[padding, padding], radius=0.04)

        # draw_title(vsk, "VALLE", padding=padding)
        
        vsk.vpype("linemerge linesimplify reloop linesort")
        # vsk.save("map/output/map_valle_7.svg")
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    MapSketch.display()
