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

# TODO: additional shapes, shape collections
# TODO: change terrain code to create islands in water, mountains, ...
# TODO: add penalty to terrain height based on distance to center

class IsoTownSketch(vsketch.SketchClass):
    
    draw_axes = vsketch.Param(False)
    draw_grid = vsketch.Param(False)
    draw_debug = vsketch.Param(False)
    draw_shading = vsketch.Param(True)
    
    amplitude = vsketch.Param(3.0)
    gain = vsketch.Param(0.2)
    octaves = vsketch.Param(3)
    falloff = vsketch.Param(0.4)
    scale = vsketch.Param(0.5)
    terrain_size = vsketch.Param(18)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.noiseDetail(self.octaves, self.falloff)
        
        if self.draw_axes: iso.draw_axes(vsk, x_axis_length=12, y_axis_length=12, scale=self.scale)
        if self.draw_grid: iso.draw_grid(vsk, x_size=12, y_size=12, scale=self.scale)
        
        shapes = []
        
        # Test all shapes:
        # shapes.append(iso.Box(1, 1, 0, 1, 1, 1, scale=self.scale))
        # shapes.append(iso.Box(0, 2, 0, 3, 1, 1, scale=self.scale))
        
        # shapes.append(iso.Slope(4, 6, 0, 1, 1, 1, rotation=iso.Rotation.XP, scale=self.scale))
        # shapes.append(iso.Slope(6, 4, 0, 1, 1, 1, rotation=iso.Rotation.YP, scale=self.scale))
        
        # shapes.append(iso.Slope(9, 7, 0, 1, 1, 1.5, rotation=iso.Rotation.XM, scale=self.scale))
        # shapes.append(iso.Slope(9, 6, 0, 1, 1, 0.5, rotation=iso.Rotation.XM, scale=self.scale))
        
        # shapes.append(iso.Slope(7, 9, 0, 1, 1, 1.5, rotation=iso.Rotation.YM, scale=self.scale))
        # shapes.append(iso.Slope(6, 9, 0, 1, 1, 0.5, rotation=iso.Rotation.YM, scale=self.scale))
        
        # shapes.append(iso.Pyramid(7, 7, 0, 1, 1, 1, scale=self.scale))
        # shapes.append(iso.Pyramid(10, 10, 0, 1, 1, 0.25, scale=self.scale))
        
        # shapes.append(iso.Stairs(4, 10, 0, 1, 1, 1, n_steps=4, rotation=iso.Rotation.XP, scale=self.scale))
        
        # shapes.append(iso.Arch(6, 12, 0, 1, 1, 1, rotation=iso.Rotation.XP, scale=self.scale))
        
        
        
        # Random thing:
        # shapes.append(iso.Box(0, 0, 0, 1, 1, 5, scale=self.scale))
        # shapes.append(iso.Box(0, 0, 5, 2, 2, 0.5, scale=self.scale))
        # shapes.append(iso.Slope(-1, 0, 0, 1, 1, 1, rotation=iso.Rotation.XP, scale=self.scale))
        # shapes.append(iso.Slope(1, 0, 0, 1, 1, 1, rotation=iso.Rotation.XM, scale=self.scale))
        # shapes.append(iso.Slope(0, -1, 0, 1, 1, 1, rotation=iso.Rotation.YP, scale=self.scale))
        # shapes.append(iso.Slope(0, 1, 0, 1, 1, 1, rotation=iso.Rotation.YM, scale=self.scale))
        
        # Terrain:
        # base terrain:
        z_grid = self.amplitude * vsk.noise(self.gain * np.linspace(0, self.terrain_size - 1, self.terrain_size),
                                            self.gain * np.linspace(0, self.terrain_size - 1, self.terrain_size))
        
        z_grid += 0.2 * self.amplitude
        
        # increase higher ranges:
        z_grid += 2 * (np.clip(z_grid, self.amplitude*0.45, np.inf) - self.amplitude*0.45)
        
        # :
        # z_grid += np.clip(20 * vsk.noise(0.15 * np.linspace(0, self.terrain_size - 1, self.terrain_size),
        #                                  0.15 * np.linspace(0, self.terrain_size - 1, self.terrain_size)), 7, np.inf) - 7
        
        
        water_level = 0.3
        ground_level = 0.4  
        
        for x, row in enumerate(z_grid):
            for y, z_top in enumerate(row):
                z_grid[x, y] -= 0.07 * ((x - 0.5 * self.terrain_size)**2 + (y - 0.5 * self.terrain_size)**2)
                if z_grid[x, y] < ground_level * self.amplitude:
                    z_grid[x, y] = water_level * self.amplitude
        
        z_tower = np.amax(z_grid)
        x_tower, y_tower = np.unravel_index(np.argmax(z_grid), z_grid.shape)
        z_grid[x_tower+1, y_tower] = z_tower
        
        for x, row in enumerate(z_grid):
            for y, z_top in enumerate(row):
                shapes.append(iso.Box(x, y, 0, 1, 1, z_top, scale=self.scale))
                
                
        # Tower:
        # # x_tower, y_tower, z_tower = (0, 0, 0)
        # z_tower = np.amax(z_grid)
        # x_tower, y_tower = np.unravel_index(np.argmax(z_grid), z_grid.shape)
        # tower_pos = [x_tower, y_tower, z_tower]
        # size_x = 0.8
        # size_y = 0.8
        # size_z = 4
        # beam_width = 0.1
        # roof_height = 1.0
        # roof_overhang = 0.2
        # shapes.append(iso.Box(*tower_pos, size_x, size_y, size_z, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos - np.array([beam_width, beam_width, 0]), beam_width, beam_width, size_z, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([size_x, -beam_width, 0]), beam_width, beam_width, size_z, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([size_x, size_y, 0]), beam_width, beam_width, size_z, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([-beam_width, size_y, 0]), beam_width, beam_width, size_z, scale=self.scale))
        
        # shapes.append(iso.Box(*tower_pos + np.array([-beam_width, -beam_width, 0]), beam_width, size_y + beam_width, beam_width, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([-beam_width, -beam_width, 0]), size_x + beam_width, beam_width, beam_width, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([-beam_width, -beam_width, size_z / 2]), beam_width, size_y + beam_width, beam_width, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([-beam_width, -beam_width, size_z / 2]), size_x + beam_width, beam_width, beam_width, scale=self.scale))
        
        # shapes.append(iso.Slope(*tower_pos + np.array([-roof_overhang, -roof_overhang, size_z]), size_x + 2*roof_overhang, size_y/2 + roof_overhang, roof_height, rotation=iso.Rotation.YP, scale=self.scale))
        # shapes.append(iso.Slope(*tower_pos + np.array([-roof_overhang, size_y/2, size_z]), size_x + 2*roof_overhang, size_y/2 + roof_overhang, roof_height, rotation=iso.Rotation.YM, scale=self.scale))
                
        # shapes.append(iso.Box(*tower_pos + np.array([-beam_width, -beam_width, 0]), size_x + beam_width, beam_width, beam_width, scale=self.scale))
        # shapes.append(iso.Stairs(*tower_pos + np.array([0, -0.4, 0]), 0.8, 0.4, 0.4, n_steps=4, rotation=iso.Rotation.YP, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([0.2, -0.1, 0.4]), 0.1, 0.1, 0.5, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([0.6, -0.1, 0.4]), 0.1, 0.1, 0.5, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([0.2, -0.1, 2.5]), 0.5, beam_width, beam_width, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([0.2, -0.1, 3.0]), 0.5, beam_width, beam_width, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([0.2, -0.1, 2.6]), beam_width, beam_width, 0.4, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([0.6, -0.1, 2.6]), beam_width, beam_width, 0.4, scale=self.scale))
        
        z_tower = np.amax(z_grid)
        x_tower, y_tower = np.unravel_index(np.argmax(z_grid), z_grid.shape)
        z_grid[x_tower+1, y_tower] = z_tower
        # x_tower, y_tower = (10, 10)
        # z_tower = z_grid[x_tower, y_tower]
        tower_pos = [x_tower+0.5, y_tower+0.2, z_tower]
        size_x = 1.0
        size_y = 0.6
        size_z = 1.0
        beam_width = 0.1
        roof_height = 0.9
        roof_overhang = 0.2
        shapes.append(iso.Box(*tower_pos, size_x, size_y, size_z, scale=self.scale))
        shapes.append(iso.Box(*tower_pos - np.array([beam_width, beam_width, 0]), beam_width, beam_width, size_z, scale=self.scale))
        shapes.append(iso.Box(*tower_pos + np.array([size_x, -beam_width, 0]), beam_width, beam_width, size_z, scale=self.scale))
        shapes.append(iso.Box(*tower_pos + np.array([size_x, size_y, 0]), beam_width, beam_width, size_z, scale=self.scale))
        shapes.append(iso.Box(*tower_pos + np.array([-beam_width, size_y, 0]), beam_width, beam_width, size_z, scale=self.scale))
        
        # shapes.append(iso.Box(*tower_pos + np.array([-beam_width, -beam_width, 0]), beam_width, size_y + beam_width, beam_width, scale=self.scale))
        # shapes.append(iso.Box(*tower_pos + np.array([-beam_width, -beam_width, 0]), size_x + beam_width, beam_width, beam_width, scale=self.scale))
        
        shapes.append(iso.Slope(*tower_pos + np.array([-roof_overhang, -roof_overhang, size_z]), size_x + 2*roof_overhang, size_y/2 + roof_overhang, roof_height, rotation=iso.Rotation.YP, scale=self.scale))
        shapes.append(iso.Slope(*tower_pos + np.array([-roof_overhang, size_y/2, size_z]), size_x + 2*roof_overhang, size_y/2 + roof_overhang, roof_height, rotation=iso.Rotation.YM, scale=self.scale))
             
        
        if self.draw_shading:
            dx_shade=0.15
            dy_shade=0.075
            dz_shade=1.0 + 1e-6
        else:
            dx_shade = dy_shade = dz_shade = None
        
        draw_order = iso.get_draw_order(shapes)
        vsk.stroke(2)
        for i in draw_order:
            shapes[i].draw(vsk, dx_shade=dx_shade, dy_shade=dy_shade, dz_shade=dz_shade)
            if self.draw_debug: shapes[i].draw_debug(vsk, offset=0.2*i)
        
        
        vsk.vpype("occult -i")
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    IsoTownSketch.display()
