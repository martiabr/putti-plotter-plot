import vsketch
import numpy as np
from enum import Enum


# What kind of tiles to do:
# - original
# - circles
# - crosses
# - double colored circles
# - squares
# - knots
# - quarter circles
# - multiple circles of different radius
# - experiment with own


class TruchetSketch(vsketch.SketchClass):
    # Sketch parameters:
    size = vsketch.Param(1.5)
    n_x = vsketch.Param(12)
    n_y = vsketch.Param(18)
    grid = vsketch.Param(True)
    
    tile_sets = Enum('TruchetTileSet', 'circles triangles diagonals')
    tile_set = vsketch.Param(tile_sets.circles.name, choices=[tile_set.name for tile_set in tile_sets])
    
    def build_circle_tiles(self):
        tiles = []
        
        tile_1 = vsketch.Vsketch()
        tile_1.detail("0.01")
        tile_1.arc(0, 0, self.size, self.size, 3*np.pi/2, 2*np.pi)
        tile_1.arc(self.size, self.size, self.size, self.size, np.pi/2, np.pi)
        tiles.append(tile_1)
        
        tile_2 = vsketch.Vsketch()
        tile_2.detail("0.01")
        tile_2.arc(self.size, 0, self.size, self.size, np.pi, 3*np.pi/2)
        tile_2.arc(0, self.size, self.size, self.size, 0, np.pi/2)
        tiles.append(tile_2)
        
        return tiles
    
    def build_tiles(self, type, show_grid):
        tiles = []
        if type == self.tile_sets.circles.name:
            tiles = self.build_circle_tiles()
        elif type == self.tile_sets.triangles.name:
            pass
        elif type == self.tile_sets.diagonals.name:
            pass
        
        if show_grid:
            for tile in tiles:
                tile.square(0, 0, self.size)
        
        return tiles
        
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.detail("0.1mm")
        
        tiles = self.build_tiles(self.tile_set, self.grid)
        n_tiles = len(tiles)
        
        # Draw grid of random tiles from chosen tile set:
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
