import numpy as np
from plotter_shapes.plotter_shapes import get_empty_sketch

def get_normal(x, y, x_next, y_next):
    dx, dy = x_next - x, y_next - y
    return np.array([-dy, dx])


def get_unit_normal(x, y, x_next, y_next):
    normal = get_normal(x, y, x_next, y_next)
    return normal / np.sqrt(normal[0]**2 + normal[1]**2)


def draw_variable_width_path(x, y, width, normals_distance=1e-1):
    sketch = get_empty_sketch()
    n = x.shape
    
    if type(width) == float:
        width = np.full(n, width)
        
    assert y.shape == n
    assert width.shape == n
    
    rest = 0.0
    normal = get_unit_normal(x[0], y[0], x[1], y[1])
    normal_next = None
    for i in range(x.shape[0] - 1):
        normal_next = get_unit_normal(x[i], y[i], x[i+1], y[i+1])
        
        dx, dy = x[i+1] - x[i], y[i+1] - y[i]
        d = np.sqrt(dx**2 + dy**2)
        
        n_normals = 0
        if d > normals_distance:  # at least one
            n_normals = int(np.floor((d - rest) / normals_distance)) + 1
            
        for j in range(n_normals):
            d_j = rest + normals_distance * j  # distance from start of segment to normal j
            x_normal_center, y_normal_center = x[i] + dx / d * d_j, y[i] + dy / d * d_j
            
            line_frac = d_j / d  # fraction along segment to normal j
            normal_j = normal + line_frac * (normal_next - normal)  # interpolate normal vector between prev normal and next normal (to avoid artifacts)
            width_j = width[i] + line_frac * (width[i+1] - width[i])
            
            sketch.line(x_normal_center - width_j * normal_j[0], y_normal_center - width_j * normal_j[1], 
                        x_normal_center + width_j * normal_j[0], y_normal_center + width_j * normal_j[1])
        # print(f"\n{i} d={d:.6f}, n_normals={n_normals}, rest={rest:.6f}")
        rest += n_normals * normals_distance - d
        
        sketch.line(x[i] + width[i] * normal[0], y[i] + width[i] * normal[1],
                    x[i+1] + width[i+1] * normal_next[0], y[i+1] + width[i+1] * normal_next[1])
        sketch.line(x[i] - width[i] * normal[0], y[i] - width[i] * normal[1],
                    x[i+1] - width[i+1] * normal_next[0], y[i+1] - width[i+1] * normal_next[1])
        
        normal = normal_next
    
    # Extra line on end segment:
    sketch.line(x[-1] - width[-1] * normal[0], y[-1] - width[-1] * normal[1], 
                x[-1] + width[-1] * normal[0], y[-1] + width[-1] * normal[1])
    return sketch
      