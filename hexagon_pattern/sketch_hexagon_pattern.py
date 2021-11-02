import vsketch
import numpy as np
# import plotter_util as plutil
# from numpy.random import uniform, randint

# TODO: make a cool border. make shaded sides with line density param.
# Redo things such that we say draw hexagon at position x,y. Then use grid for drawing. And use with vsk.pushMatrix(): for easy transforms

def get_circle_points(R):
    points = []
    for x in range(-R,R+1):
        Y = int((R**2-x**2)**0.5) # bound for y given x
        for y in range(-Y,Y+1):
            points.append([x,y])
    return np.array(points)

def get_hexagon_points():
    return None

def get_square_points(width, height, zig_zag=False):
    points = []
    for y in range(height):
        for x in range(width):
            if not (x == 0 and y % 2 == 0 and not zig_zag):
                points.append([x,y])
    return np.array(points)
        
def get_points(size_1=None, size_2=None, border_type='circle'):
    if border_type == 'circle':
        return get_circle_points(size_1)
    elif border_type == 'hexagon':
        return get_hexagon_points()
    elif border_type == 'square':
        return get_square_points(size_1, size_2)

def draw_hexagon_cube(vsk, x, y, size, angle_rad, shading=True, n_shading_left=12, n_shading_right=6):
    with vsk.pushMatrix():
        vsk.translate(x, y)
        # vsk.circle(0, 0, 0.2)
        vsk.line(0, 0, 0, size)
        vsk.line(0, 0, size * np.cos(angle_rad), size * np.sin(angle_rad))
        vsk.line(size * np.cos(angle_rad), size * np.sin(angle_rad), 2 * size * np.cos(angle_rad), 0)
        vsk.line(0, 0, size * np.cos(angle_rad), -size * np.sin(angle_rad))
        vsk.line(size * np.cos(angle_rad), -size * np.sin(angle_rad), 2 * size * np.cos(angle_rad), 0)
        
        vsk.line(0, size, size * np.cos(angle_rad), size * np.sin(angle_rad) + size)
        vsk.line(size * np.cos(angle_rad), size * np.sin(angle_rad) + size, 2 * size * np.cos(angle_rad), size)
        
        if shading:
            for i in range(n_shading_left):
                y = i / n_shading_left * size
                vsk.line(0, y, size * np.cos(angle_rad), y + size * np.sin(angle_rad))
                
            for i in range(n_shading_right):
                y = i / n_shading_right * size
                vsk.line(size * np.cos(angle_rad), y + size * np.sin(angle_rad), 2 * size * np.cos(angle_rad), y)
                
        
        vsk.translate(size * np.cos(angle_rad), size * np.sin(angle_rad))
        
        vsk.line(0, 0, 0, size)
        
        vsk.translate(size * np.cos(angle_rad), -size * np.sin(angle_rad))
        
        vsk.line(0, 0, 0, size)
    
class HexagonPatternSketch(vsketch.SketchClass):
    # Sketch parameters:
    
    angle = vsketch.Param(30.0, step=1.0)
    size = vsketch.Param(1.0, min_value=0.0)
    
    n_y = vsketch.Param(12, min_value=1)
    n_x = vsketch.Param(6, min_value=1)
    
    n_shading_left = vsketch.Param(12, min_value=1)
    n_shading_right = vsketch.Param(6, min_value=1)
    
    width = 21.0
    height = 29.7
    
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        # points = get_points(int(self.n_x / 2), border_type='circle')
        points = get_points(self.n_x, self.n_y, border_type='square')
        
        angle_rad = np.deg2rad(self.angle)
        
        for point in points:
            parity_offset = 0.5*(point[1] % 2) - 0.25
            x = (point[0] + parity_offset) * self.size * 2 * np.cos(angle_rad)
            y = self.size * (1 + np.sin(angle_rad)) * point[1]
            draw_hexagon_cube(vsk, x, y, self.size, angle_rad, n_shading_left=self.n_shading_left, n_shading_right=self.n_shading_right)
                
            # vsk.translate(-2 * (self.n_x + parity_offset) * self.size * np.cos(angle_rad),
            #               self.size + self.size * np.sin(angle_rad))
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    HexagonPatternSketch.display()
