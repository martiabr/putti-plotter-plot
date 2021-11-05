import vsketch
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp2d, griddata
from scipy.ndimage import gaussian_filter

class MapSketch(vsketch.SketchClass):
    # Sketch parameters:
    # radius = vsketch.Param(2.0)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        scale = 0.0035
        
        vsk.rotate(np.pi/2)
        
        coords = np.load('map/coords_bymarka_2.npy')
        print('Data loaded with shape', coords.shape)
        
        # Fix water levels:
        z_min = 1.0
        for i, point in enumerate(coords):
            if point[2] < z_min:
                coords[i,2] = z_min

        # Generate a regular grid to interpolate the data:
        N = 2000
        # xmin = np.min(coords[:,0])
        # xmax = np.max(coords[:,0])
        # ymin = np.min(coords[:,1])
        # ymax = np.max(coords[:,1])
        xmin = 561400
        xmax = 568300
        ymin = 7031400
        ymax = 7036400
        xi = np.linspace(xmin, xmax, N)
        yi = np.linspace(ymin, ymax, N)

        # Interpolate using delaunay triangularization :
        zi = griddata((coords[:,0], coords[:,1]),coords[:,2], (xi[None,:], yi[:,None]), method='nearest')
        print('Interpolation completed.')

        # Filter using Gaussian filter:
        sigma = 4.5 * np.ones(2)
        z_smooth = gaussian_filter(zi, sigma)
        # TODO: test other smoothers
        print('Smoothing completed.')
        
        levels_final = 80
        
        cs = plt.contour(xi, yi, z_smooth, levels=levels_final, colors='k', linewidths=0.5)
        plt.gca().axis('off')

        paths = cs.allsegs
        for i in range(len(paths)):
            for j in range(len(paths[i])):
                for k in range(len(paths[i][j])-1):
                    vsk.line(scale*paths[i][j][k][0], scale*paths[i][j][k][1], scale*paths[i][j][k+1][0], scale*paths[i][j][k+1][1])
        print('done')
        
        # TODO: text
        # vsk.vpype("text -f futural -s 20 -p 0 1000 bymarka2010")
        
        vsk.vpype("linemerge linesimplify reloop linesort")
        vsk.save("map/output/map_bymarka_2_optimized_3.svg")
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    MapSketch.display()
