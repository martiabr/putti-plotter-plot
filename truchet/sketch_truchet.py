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

# TODO:
# - make it possible to choose distirbution of tiles explicitly
# - possibly shadows


class TruchetSketch(vsketch.SketchClass):
    # Sketch parameters:
    size = vsketch.Param(1.5, decimals=2)
    n_x = vsketch.Param(12)
    n_y = vsketch.Param(18)
    n_fill = vsketch.Param(10)
    grid = vsketch.Param(False)
    
    tile_sets = Enum('TruchetTileSet', 'circles triangles diagonals knot')
    tile_set = vsketch.Param(tile_sets.knot.name, choices=[tile_set.name for tile_set in tile_sets])
    
    knot_mode = vsketch.Param('only_center', choices=['only_center', 'only_outer', 'many'])
    knot_N = vsketch.Param(4)
    knot_thickness = vsketch.Param(0.25)
    do_skip = vsketch.Param(False)
    knot_end_mode = vsketch.Param('soft', choices=['none', 'hard', 'soft', 'continue'])
    
    def init_tiles(self, n, detail='0.01'):
        tiles = []
        
        for i in range(n):
            tile = vsketch.Vsketch()
            tile.detail(detail)
            tiles.append(tile)
        
        return tiles
    
    def build_circle_tiles(self):
        tiles = self.init_tiles(2)
        
        tiles[0].arc(0, 0, self.size, self.size, 3*np.pi/2, 2*np.pi)
        tiles[0].arc(self.size, self.size, self.size, self.size, np.pi/2, np.pi)
        
        tiles[1].arc(self.size, 0, self.size, self.size, np.pi, 3*np.pi/2)
        tiles[1].arc(0, self.size, self.size, self.size, 0, np.pi/2)
        
        return tiles
    
    # def build_circle_tiles(self):
    #     tiles = self.init_tiles(4)
        
    #     n = 5
    #     for i in range(1, n + 1):
    #         tiles[0].arc(0, 0, self.size, self.size, 3*np.pi/2, 2*np.pi)
    #         tiles[0].arc(self.size, self.size, self.size, self.size, np.pi/2, np.pi)
            
    #         tiles[1].arc(self.size, 0, self.size, self.size, np.pi, 3*np.pi/2)
    #         tiles[1].arc(0, self.size, self.size, self.size, 0, np.pi/2)
            
    #         radius = self.size * i / n
            
    #         tiles[2].arc(0, 0, radius, radius, 3*np.pi/2, 2*np.pi)
    #         tiles[2].arc(self.size, self.size, radius, radius, np.pi/2, np.pi)
            
    #         tiles[3].arc(self.size, 0, radius, radius, np.pi, 3*np.pi/2)
    #         tiles[3].arc(0, self.size, radius, radius, 0, np.pi/2)
        
    #     return tiles
    
    def build_knot_arcs(self, tiles, dx, do_skip=False, skip=0.0):
        dr = 2 * dx
        
        tiles[0].arc(0, 0, self.size + dr, self.size + dr, 3*np.pi/2, 2*np.pi)
        tiles[0].arc(self.size, self.size, self.size + dr, self.size + dr, np.pi/2, np.pi)
        
        tiles[1].arc(self.size, 0, self.size + dr, self.size + dr, np.pi, 3*np.pi/2)
        tiles[1].arc(0, self.size, self.size + dr, self.size + dr, 0, np.pi/2)
        
        if do_skip:
            tiles[2].line(self.size / 2 + dx, 0, self.size / 2 + dx, self.size / 2 - skip)
            tiles[2].line(self.size / 2 + dx, self.size / 2 + skip, self.size / 2 + dx, self.size)
            tiles[2].line(0, self.size / 2 + dx, self.size, self.size / 2 + dx)
            
            tiles[3].line(self.size / 2 + dx, 0, self.size / 2 + dx, self.size)
            tiles[3].line(0, self.size / 2 + dx, self.size / 2 - skip, self.size / 2 + dx)
            tiles[3].line(self.size / 2 + skip, self.size / 2 + dx, self.size, self.size / 2 + dx)

        else:
            tiles[2].line(self.size / 2 + dx, 0, self.size / 2 + dx, self.size)
            tiles[2].line(0, self.size / 2 + dx, self.size, self.size / 2 + dx)
            
            tiles[3].line(self.size / 2 + dx, 0, self.size / 2 + dx, self.size)
            tiles[3].line(0, self.size / 2 + dx, self.size, self.size / 2 + dx)
        
        return tiles
    
    def build_knot_tiles(self, mode):
        tiles = self.init_tiles(4)
        
        # TODO: either only center, only outer lines or loop of lines
        # TODO: with or without "3D"
        # TODO: function of arcs, getting ugly
        
        if mode == 'only_center':
            tiles = self.build_knot_arcs(tiles, 0.0)
        elif mode == 'only_outer':
            tiles = self.build_knot_arcs(tiles, self.knot_thickness, self.do_skip, self.knot_thickness)
            tiles = self.build_knot_arcs(tiles, -self.knot_thickness, self.do_skip, self.knot_thickness)
        elif mode == 'many':
            tiles = self.build_knot_arcs(tiles, 0.0, self.do_skip, self.knot_thickness)
            for i in range(1, self.knot_N + 1):
                dx = self.knot_thickness * i / self.knot_N
                tiles = self.build_knot_arcs(tiles, dx, self.do_skip, self.knot_thickness)
                tiles = self.build_knot_arcs(tiles, -dx, self.do_skip, self.knot_thickness)
        
        return tiles
    
    def add_knot_ends(self, vsk, mode='hard'):
        with vsk.pushMatrix():
            for y in range(self.n_y):
                vsk.translate(0, -self.size)
                if mode == 'hard':
                    vsk.line(0, self.size / 2 - self.knot_thickness, 0, self.size / 2 + self.knot_thickness)
                    vsk.line(self.size * self.n_x, self.size / 2 - self.knot_thickness, self.size * self.n_x, self.size / 2 + self.knot_thickness)
                elif mode == 'soft':
                    vsk.arc(0, self.size / 2, 2 * self.knot_thickness, 2 * self.knot_thickness, np.pi / 2, 3 * np.pi / 2)
                    vsk.arc(self.size * self.n_x, self.size / 2, 2 * self.knot_thickness, 2 * self.knot_thickness, -np.pi / 2, np.pi / 2)
                elif mode == 'continue' and y % 2 == 1:
                    vsk.arc(0, self.size, self.size - 2 * self.knot_thickness, self.size - 2 * self.knot_thickness, np.pi / 2, 3 * np.pi / 2)
                    vsk.arc(0, self.size, self.size + 2 * self.knot_thickness, self.size + 2 * self.knot_thickness, np.pi / 2, 3 * np.pi / 2)
                    vsk.arc(self.size * self.n_x, self.size, self.size - 2 * self.knot_thickness, self.size - 2 * self.knot_thickness, -np.pi / 2, np.pi / 2)
                    vsk.arc(self.size * self.n_x, self.size, self.size + 2 * self.knot_thickness, self.size + 2 * self.knot_thickness, -np.pi / 2, np.pi / 2)
                    
                                    
        with vsk.pushMatrix():
            for x in range(self.n_x):
                if mode == 'hard':
                    vsk.line(self.size / 2 - self.knot_thickness, 0, self.size / 2 + self.knot_thickness, 0)
                    vsk.line(self.size / 2 - self.knot_thickness, -self.size * self.n_y, self.size / 2 + self.knot_thickness, -self.size * self.n_y)
                elif mode == 'soft':
                    vsk.arc(self.size / 2, 0, 2 * self.knot_thickness, 2 * self.knot_thickness, np.pi, 0)
                    vsk.arc(self.size / 2, -self.size * self.n_y, 2 * self.knot_thickness, 2 * self.knot_thickness, 0, np.pi)
                elif mode == 'continue' and x % 2 == 0:
                    vsk.arc(self.size, 0, self.size - 2 * self.knot_thickness, self.size - 2 * self.knot_thickness, np.pi, 0)
                    vsk.arc(self.size, 0, self.size + 2 * self.knot_thickness, self.size + 2 * self.knot_thickness, np.pi, 0)
                    vsk.arc(self.size, -self.size * self.n_y, self.size - 2 * self.knot_thickness, self.size - 2 * self.knot_thickness, 0, np.pi)
                    vsk.arc(self.size, -self.size * self.n_y, self.size + 2 * self.knot_thickness, self.size + 2 * self.knot_thickness, 0, np.pi)
                    
                vsk.translate(self.size, 0)
    
    def build_triangles_tiles(self):
        tiles = self.init_tiles(4)
        
        for i in range(self.n_fill):
            size = self.size * i / self.n_fill
            tiles[0].line(size, 0, size, size)
            tiles[1].line(size, self.size - size, size, 0)
            tiles[2].line(size, size, size, self.size)
            tiles[3].line(size, self.size, size, self.size - size)
            
        tiles[0].polygon([(0, 0), (self.size, 0), (self.size, self.size)], close=True)
        tiles[1].polygon([(0, 0), (0, self.size), (self.size, 0)], close=True)
        tiles[2].polygon([(0, 0), (0, self.size), (self.size, self.size)], close=True)
        tiles[3].polygon([(0, self.size), (self.size, self.size), (self.size,0)], close=True)
        
        return tiles
    
    def build_tiles(self, type, show_grid):
        tiles = []
        if type == self.tile_sets.circles.name:
            tiles = self.build_circle_tiles()
        elif type == self.tile_sets.triangles.name:
            tiles = self.build_triangles_tiles()
        elif type == self.tile_sets.diagonals.name:
            pass
        elif type == self.tile_sets.knot.name:
            tiles = self.build_knot_tiles(self.knot_mode)
        
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
        
        # vsk.fill(2)
        # vsk.penWidth("1mm", 2)
        # vsk.polygon([(0,0), (self.size,0), (self.size, self.size)], close=True)
        
        # Draw grid of random tiles from chosen tile set:
        for y in range(self.n_y):
            with vsk.pushMatrix():
                for x in range(self.n_x):
                    tile_index = np.random.randint(0, n_tiles)
                    vsk.sketch(tiles[tile_index])
                    vsk.translate(self.size, 0)
            vsk.translate(0, self.size)
        
        if self.tile_set == self.tile_sets.knot.name and self.knot_end_mode != 'none':
            self.add_knot_ends(vsk, self.knot_end_mode)
            
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    TruchetSketch.display()
