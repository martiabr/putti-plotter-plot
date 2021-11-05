import vsketch
import numpy as np

class TruchetSketch(vsketch.SketchClass):
    # Sketch parameters:
    size = vsketch.Param(1.5)
    n_x = vsketch.Param(12)
    n_y = vsketch.Param(18)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.detail("0.1mm")
        
        # create tiles:
        # TODO: move to different functions that build tiles and pick which with dropdown
        tile_1 = vsketch.Vsketch()
        tile_1.detail("0.01")
        tile_1.square(0, 0, self.size)
        tile_1.arc(0, 0, self.size, self.size, 3*np.pi/2, 2*np.pi)
        tile_1.arc(self.size, self.size, self.size, self.size, np.pi/2, np.pi)
        
        tile_2 = vsketch.Vsketch()
        tile_2.detail("0.01")
        tile_2.square(0, 0, self.size)
        tile_2.arc(self.size, 0, self.size, self.size, np.pi, 3*np.pi/2)
        tile_2.arc(0, self.size, self.size, self.size, 0, np.pi/2)
        
        tiles = [tile_1, tile_2]
        n_tiles = len(tiles)
        
        for y in range(self.n_y):
            with vsk.pushMatrix():
                for x in range(self.n_x):
                    tile_index = np.random.randint(0, n_tiles)
                    vsk.sketch(tiles[tile_index])
                    vsk.translate(self.size, 0)
            vsk.translate(0, self.size)
            

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    TruchetSketch.display()
