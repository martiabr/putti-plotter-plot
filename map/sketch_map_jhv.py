import vsketch
from matplotlib import pyplot as plt
import numpy as np
import rasterio as rio
from scipy.ndimage import gaussian_filter

class MapSketch(vsketch.SketchClass):

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        # vsk.detail("1mm")
        vsk.penWidth("2mm")
        
        img_1 = rio.open('map/jotunheimen_vest/dtm10/data/dtm10_6801_4_10m_z33.tif').read(1)
        img_2 = rio.open('map/jotunheimen_vest/dtm10/data/dtm10_6801_3_10m_z33.tif').read(1)
        img = np.vstack((img_1[:-50], img_2))  # functionify
        
        scale = 17.7 / img.shape[1]
        vsk.scale(scale, scale)  # functionify
                
        sigma = 1.5 * np.ones(2)
        img = gaussian_filter(img, sigma)  # functionify

        x_0 = 0
        y_0 = 2350
        x_1 = 5044
        y_1 = 9700
        img = img[y_0:y_1, x_0:x_1]  # functionify

        levels = 35

        plt.figure(figsize=(2*2.1,2*2.97))
        cs = plt.contour(img, levels=levels, colors='k', linewidths=0.2)
        plt.gca().invert_yaxis()
        plt.axis('equal')

        paths = cs.allsegs
        for i in range(len(paths)):
            for j in range(len(paths[i])):
                for k in range(len(paths[i][j])-1):
                    vsk.line(paths[i][j][k][0], paths[i][j][k][1], paths[i][j][k+1][0], paths[i][j][k+1][1])
        print('done')
        
        # Text:
        vsk.vpype("text -f rowmans -s 20 -p 17.75cm 26.4cm --align right \"JOTUNHEIMEN VEST\"")
        
        vsk.vpype("linemerge linesimplify reloop linesort")
        vsk.save("map/output/map_jhv_2.svg")
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    MapSketch.display()
