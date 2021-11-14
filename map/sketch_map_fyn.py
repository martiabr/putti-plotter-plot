import vsketch
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter

class MapSketch(vsketch.SketchClass):

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        # vsk.detail("1mm")
        vsk.penWidth("2mm")
        
        scale = 0.031
        
        path = "map/fyn/"
        file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        sea_level = -0.2
        all_data = np.zeros((4, 1000, 1000))
        for i, file_name in enumerate(file_names):
            data = np.loadtxt(path + file_name, skiprows=6)
            
            for x,y in np.ndindex(data.shape):  # fix sea level
                if data[x, y] < sea_level:
                    data[x,y] = sea_level
            
            all_data[i] = data

        data = np.zeros((2000, 2000))
        data[:1000, :1000] = all_data[1]
        data[1000:, :1000] = all_data[0]
        data[:1000, 1000:] = all_data[3]
        data[1000:, 1000:] = all_data[2]

        sigma = 1.0 * np.ones(2)
        data_smooth = gaussian_filter(data, sigma)

        ##############

        levels = 35

        x_0 = 910
        y_0 = 450
        x_1 = 1540
        y_1 = 1310

        data_smooth = data_smooth[y_0:y_1, x_0:x_1]

        plt.figure(figsize=(2*2.1,2*2.97))
        cs = plt.contour(data_smooth, levels=levels, colors='k', linewidths=0.2)
        plt.gca().invert_yaxis()
        plt.axis('equal')

        paths = cs.allsegs
        for i in range(len(paths)):
            for j in range(len(paths[i])):
                for k in range(len(paths[i][j])-1):
                    vsk.line(scale*paths[i][j][k][0], scale*paths[i][j][k][1], scale*paths[i][j][k+1][0], scale*paths[i][j][k+1][1])
        print('done')
        
        # TODO: text
        # vsk.vpype("text -f futural -s 20 -p 0 1000 bymarka2010")
        
        vsk.vpype("linemerge linesimplify reloop linesort")
        vsk.save("map/output/map_fyn.svg")
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    MapSketch.display()
