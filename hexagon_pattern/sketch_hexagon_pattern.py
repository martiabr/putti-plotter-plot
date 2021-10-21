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

def get_square_points():
    return None
        
def get_points(size, border_type='circle'):
    if border_type == 'circle':
        return get_circle_points(size)
    elif border_type == 'hexagon':
        return get_hexagon_points()
    elif border_type == 'square':
        return get_square_points()

def draw_hexagon_cube(vsk, x, y, size, angle_rad):
    vsk.line(0, 0, 0, size)
    vsk.line(0, 0, size * np.cos(angle_rad), size * np.sin(angle_rad))
    vsk.line(size * np.cos(angle_rad), size * np.sin(angle_rad), 2 * size * np.cos(angle_rad), 0)
    vsk.line(0, 0, size * np.cos(angle_rad), -size * np.sin(angle_rad))
    vsk.line(size * np.cos(angle_rad), -size * np.sin(angle_rad), 2 * size * np.cos(angle_rad), 0)
    
    vsk.line(0, size, size * np.cos(angle_rad), size * np.sin(angle_rad) + size)
    vsk.line(size * np.cos(angle_rad), size * np.sin(angle_rad) + size, 2 * size * np.cos(angle_rad), size)
    
    vsk.translate(size * np.cos(angle_rad), size * np.sin(angle_rad))
    
    vsk.line(0, 0, 0, size)
    
    vsk.translate(size * np.cos(angle_rad), -size * np.sin(angle_rad))
    
    vsk.line(0, 0, 0, size)
    
class HexagonPatternSketch(vsketch.SketchClass):
    # Sketch parameters:
    
    angle = vsketch.Param(30.0, step=1.0)
    size = 1.0
    
    n_y = vsketch.Param(10, min_value=1)
    n_x = vsketch.Param(10, min_value=1)
    
    width = 21
    height = 29.7
    

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        points = get_points(int(self.n_x / 2))
        print(points)
        
        angle_rad = np.deg2rad(self.angle)
        
        for point in points:
            draw_hexagon_cube(vsk, 0, 0, self.size, angle_rad)
                
            parity_offset = (point[1] % 2) - 0.5
            vsk.translate(-2 * (self.n_x + parity_offset) * self.size * np.cos(angle_rad), self.size + self.size * np.sin(angle_rad))
        

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    HexagonPatternSketch.display()
