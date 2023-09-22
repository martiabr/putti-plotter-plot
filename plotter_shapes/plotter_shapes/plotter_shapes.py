import random
import vsketch
import numpy as np
from shapely import Polygon, affinity, Point, MultiPoint

def pick_random_element(probs):
    return np.random.choice(len(probs), p=probs)


def enum_type_to_int(enum_type):
    return enum_type.value - 1


def get_empty_sketch(detail=1e-2):
    sketch = vsketch.Vsketch()
    sketch.detail(detail)
    return sketch


def draw_circle(x, y, radius):  
    sketch = get_empty_sketch()
    sketch.circle(x, y, radius=radius)
    return sketch


def draw_filled_circle(x, y, radius, line_width=1e-2):
    sketch = get_empty_sketch()
    N = int(radius / line_width)
    for r in np.linspace(radius, 0, N):
        sketch.circle(x, y, radius=r)
    return sketch

        
def draw_shaded_circle(x_0, y_0, radius, fill_distance, angle=0.0, fill_gain=1.0):
    sketch = get_empty_sketch()
    sketch.circle(x_0, y_0, radius=radius)
    # N = np.max((0, int(np.round(2 * (radius / fill_distance - 1)))))
    N = np.max((0, int(np.round(fill_gain * 2 * (radius / fill_distance - 1)))))
    fill_distance = fill_gain * 2 * radius / (N + 1)
    start = -radius + fill_distance
    end = -start
    fill_end = start + fill_gain * (end - start)
    sketch.translate(x_0, y_0)
    sketch.rotate(angle)
    for x in np.linspace(start, fill_end, N, endpoint=True):
        dy = radius * np.sin(np.arccos(x / radius))
        sketch.line(x, -dy, x, dy) 
    return sketch


def draw_dot_shaded_circle(x_0, y_0, radius, dot_distance, vsk=None, noise_gain=None, noise_freq=1.0):
    if noise_gain is not None: vsk.noiseSeed(np.random.randint(1e6))
    sketch = get_empty_sketch()
    sketch.translate(x_0, y_0)
    sketch.circle(0, 0, radius=radius)
    sketch.translate(-radius, -radius)
    N = np.max((0, int(np.round(2 * (radius / dot_distance - 1)))))
    dot_distance = 2 * radius / (N + 1)
    for x in np.linspace(0, 2 * radius, N + 2, endpoint=True):
        for y in np.linspace(0, 2 * radius, N + 2, endpoint=True):
            x_noise, y_noise = 0.0, 0.0
            if noise_gain is not None and vsk is not None:
                angle = 2 * np.pi * vsk.noise(noise_freq * x, noise_freq * y)
                x_noise = noise_gain * dot_distance * np.cos(angle)
                y_noise = noise_gain * dot_distance * np.sin(angle)
            x_total, y_total = x + x_noise, y + y_noise
            if (x_total - radius)**2 +(y_total - radius)**2 < radius**2:
                sketch.circle(x_total, y_total, radius=1e-3)
    return sketch


def draw_partial_filled_circle(x, y, radius, fill_gain=0.5, fill_distance=1e-2):
    return draw_shaded_circle(x, y, radius, fill_distance=fill_distance, fill_gain=fill_gain)
     

def draw_shaded_rect(x, y, width, height, fill_distance, angle=0.0):
    sketch = get_empty_sketch()
    sketch.rect(x, y, width, height, mode="center")
    N = np.max((0, int(np.round(height / fill_distance - 1))))
    fill_distance = height / (N + 1)
    sketch.translate(x, y)
    for y in np.linspace(-0.5*height + fill_distance, 0.5*height - fill_distance, N, endpoint=True):
        sketch.line(-0.5*width, y, 0.5*width, y)
    
    # TODO: add arbitrary shading angle
    # fill_distance_y = fill_distance * np.sin(angle)
    # N_y = np.max((0, int(np.round(height / fill_distance_y - 1))))
    # for y in np.linspace(-0.5*height + fill_distance_y, 0.5*height - fill_distance_y, N_y, endpoint=True):
    #     x = (y + 0.5*height) / np.tan(angle) - 0.5*width
    #     sketch.line(x, -0.5*height, -0.5*width, y)
    return sketch


def draw_dot_shaded_rect(x_0, y_0, width, height, dot_distance, vsk=None, noise_gain=None, noise_freq=1.0):
    if noise_gain is not None: vsk.noiseSeed(np.random.randint(1e6))
    sketch = get_empty_sketch()
    sketch.translate(x_0, y_0)
    sketch.rect(0, 0, width, height, mode="center")
    sketch.translate(-0.5 * width, -0.5 * height)
    N_x = np.max((0, int(np.round(width / dot_distance - 1))))
    N_y = np.max((0, int(np.round(height / dot_distance - 1))))
    dot_distance_x = width / (N_x + 1)
    dot_distance_y = height / (N_y + 1)
    for x in np.linspace(0, width, N_x + 2, endpoint=True):
        for y in np.linspace(0, height, N_y + 2, endpoint=True):
            x_noise, y_noise = 0.0, 0.0
            if noise_gain is not None and vsk is not None:
                angle = 2 * np.pi * vsk.noise(noise_freq * x, noise_freq * y)
                x_noise = noise_gain * dot_distance_x * np.cos(angle)
                y_noise = noise_gain * dot_distance_y * np.sin(angle)
            x_total, y_total = x + x_noise, y + y_noise
            if x_total > 0.0 and x_total < width and y_total > 0.0 and y_total < height:
                sketch.circle(x_total, y_total, radius=1e-3)
    return sketch
    

def draw_filled_rect(x, y, width, height, angle=0.0):
    sketch = draw_shaded_rect(x, y, width, height, fill_distance=1e-2, angle=angle)
    return sketch


def draw_rect(x, y, width, height):
    sketch = get_empty_sketch()
    sketch.rect(x, y, width, height, mode="center")
    return sketch


def draw_triangle(x, y, width, height):
    sketch = get_empty_sketch()
    sketch.triangle(x - 0.5 * width, y + 0.5 * height, x + 0.5 * width, y + 0.5 * height, x, y - 0.5 * height)
    return sketch


def draw_shaded_triangle(x, y, width, height, fill_distance):
    sketch = draw_triangle(x, y, width, height)
    N = np.max((0, int(np.round(width / fill_distance - 1))))
    fill_distance = width / (N + 1)
    sketch.translate(x, y)
    for x in np.linspace(-0.5 * width + fill_distance, 0.5 * width - fill_distance, N, endpoint=True):
        sketch.line(x, 0.5 * height, x, - 0.5 * height + np.abs(x) * 2 * height / width)
    return sketch
    
    
def draw_filled_triangle(x, y, width, height):
    sketch = draw_shaded_triangle(x, y, width, height, fill_distance=1e-2)
    return sketch


def draw_dot_circle(x, y, radius, radius_inner):
    sketch = get_empty_sketch()
    sketch.circle(x, y, radius=radius)
    sketch.sketch(draw_filled_circle(x, y, radius=radius_inner))
    return sketch


def draw_pole(x, y, pole_width, pole_height, radius):
    sketch = get_empty_sketch()
    sketch.translate(x, y -0.5 * pole_height)
    sketch.rect(0, 0, pole_width, pole_height, mode="center")
    sketch.translate(0, -0.5 * (pole_height + radius))
    sketch.circle(0, 0, radius=radius)
    sketch.translate(0, pole_height + 0.5 * radius)
    return sketch

def draw_cross(x, y, size):
    sketch = get_empty_sketch()
    sketch.line(x - 0.5 * size, y, x + 0.5 * size, y)
    sketch.line(x, y - 0.5 * size, x, y + 0.5 * size)
    return sketch

def draw_asterix(x, y, size):
    sketch = get_empty_sketch()
    sketch.translate(x, y)
    for i in range(3):
        sketch.line(-0.5 * size, 0, 0.5 * size, 0)
        sketch.rotate(np.pi / 3)
    return sketch
    
def draw_flag(x, y, pole_width, pole_height, flag_width, flag_height, right=True, triangular=False):
    sketch = get_empty_sketch()
    sketch.translate(x, y - 0.5 * pole_height)
    sketch.rect(0, 0, pole_width, pole_height, mode="center")
    sketch.translate(0, 0.5 * (flag_height - pole_height))

    if triangular:
        sketch.translate(0, -0.5 * flag_height)
        if right:
            sketch.translate(0.5 * pole_width, 0)
            sketch.triangle(0, 0, flag_width, 0.5 * flag_height, 0, flag_height)
        else:
            sketch.translate(-0.5 * pole_width, 0)
            sketch.triangle(0, 0, -flag_width, 0.5 * flag_height, 0, flag_height)
    else:
        if right:
            sketch.translate(0.5 * (flag_width + pole_width), 0)
        else:
            sketch.translate(-0.5 * (flag_width + pole_width), 0)
        sketch.rect(0, 0, flag_width, flag_height, mode="center")
        
    return sketch


def draw_line(x, y, length):
    sketch = get_empty_sketch()
    sketch.line(x, y, x, y - length)
    return sketch


def draw_dashed_line(x_0, y_0, x, y, dash_size=1e-1, factor=0.5):
    sketch = get_empty_sketch()
    total_dist = np.sqrt((x - x_0)**2 + (y - y_0)**2)
    theta = np.arctan2(y - y_0, x - x_0)
    intervals = int(total_dist / dash_size)
    actual_dash_size = factor * 0.5 * total_dist / (intervals + 0.5)
    for dist in np.linspace(0, total_dist - actual_dash_size, intervals):
        sketch.line(x_0 + dist * np.cos(theta), y_0 + dist * np.sin(theta), x_0 + (dist + actual_dash_size) * np.cos(theta), 
                    y_0 + (dist + actual_dash_size) * np.sin(theta))
    return sketch
    

def draw_thick_line(x, y, length, width=1e-2):
    return draw_filled_rect(x, y - 0.5 * length, width, length)


def sample_random_points_on_circle(radius, N=1):
    theta_i = np.random.uniform(0, 2 * np.pi, N).squeeze()
    radius_i = radius * np.sqrt(np.random.uniform(0, 1.0, N).squeeze())
    x_i, y_i = radius_i * np.array([np.cos(theta_i), np.sin(theta_i)])
    return x_i, y_i


def draw_speckled_shaded_circle(x_0, y_0, radius, density):
    sketch = get_empty_sketch()
    sketch.translate(x_0, y_0)
    sketch.circle(0, 0, radius=radius)
    
    N_speckles = int(np.round(density * np.pi * radius**2))
    for i in range(N_speckles):
        x_i, y_i = sample_random_points_on_circle(radius)
        sketch.circle(x_i, y_i, radius=1e-4)
    return sketch


def draw_speckled_shaded_rect(x_0, y_0, width, height, density):
    sketch = get_empty_sketch()
    sketch.translate(x_0, y_0)
    sketch.rect(0, 0, width, height, mode="center")
    sketch.translate(-0.5 * width, -0.5 * height)
    
    N_speckles = int(np.round(density * width * height))
    for i in range(N_speckles):
        x_i = np.random.uniform(0, width)
        y_i = np.random.uniform(0, height)
        sketch.circle(x_i, y_i, radius=1e-4)
    return sketch



def draw_dash_shading(length, padding, N_tries, f_sample_point, sketch, N_allowed_fails=100):
    rects = []
    last_valid = 0
    for i in range(N_tries):
        if i - last_valid > N_allowed_fails:  # termination criteria
            break
        
        x_i, y_i = f_sample_point()
        
        candidate_rect = Polygon(np.array([[x_i + 0.5 * length + padding, y_i + padding], 
                                           [x_i + 0.5 * length + padding, y_i - padding],
                                           [x_i - 0.5 * length - padding, y_i + padding], 
                                           [x_i - 0.5 * length - padding, y_i - padding]]))
        theta = np.random.uniform(0, np.pi)
        candidate_rect = affinity.rotate(candidate_rect, theta, use_radians=True)
        
        valid = True
        for rect in rects:
            if candidate_rect.intersects(rect):
                valid = False
                break
        if valid:
            with sketch.pushMatrix():
                sketch.translate(x_i, y_i)
                sketch.rotate(theta)
                sketch.line(-0.5 * length, 0, 0.5 * length, 0)
                rects.append(candidate_rect)
    return sketch


def draw_dash_shaded_rect(x_0, y_0, width, height, dash_length_gain=0.1, 
                          padding_gain=0.03, N_tries=3000, N_allowed_fails=500):
    sketch = get_empty_sketch() 
    sketch.translate(x_0, y_0)  
    sketch.rect(0, 0, width, height, mode="center")
    
    mean_side_length = 0.5 * (width + height)
    length = 0.5 * dash_length_gain * mean_side_length  # 0.5 here to normalize relative to circle function
    padding = 0.5 * padding_gain * mean_side_length
    
    def sample_random_point_in_rect():
        x = np.random.uniform(0.5 * (length - width), 0.5 * (width - length))
        y = np.random.uniform(0.5 * (length - height), 0.5 * (height - length))
        return x, y
    
    sketch = draw_dash_shading(length, padding, N_tries, f_sample_point=sample_random_point_in_rect, sketch=sketch,
                               N_allowed_fails=N_allowed_fails)
    return sketch


def draw_dash_shaded_circle(x_0, y_0, radius, dash_length_gain=0.1, padding_gain=0.03, 
                            N_tries=3000, N_allowed_fails=500):
    sketch = get_empty_sketch()
    sketch.translate(x_0, y_0)
    sketch.circle(0, 0, radius=radius)
    
    length = dash_length_gain * radius
    padding = padding_gain * radius
    
    lambda_sample_point= lambda: sample_random_points_on_circle(radius - 0.5 * length)
    sketch = draw_dash_shading(length, padding, N_tries, f_sample_point=lambda_sample_point, sketch=sketch,
                               N_allowed_fails=N_allowed_fails)
    return sketch



def draw_dot_evenly_shading(f_sample_points, sketch, N_points, radius):
    xs, ys = f_sample_points(N_points)
    points = MultiPoint([(x, y) for (x, y) in zip(xs, ys)])
    # print(points)
    chosen_points = []
    
    # Sample bunch of random points, sample random, remove all inside radius, repeat until none left.
    while (type(points) == MultiPoint and len(points.geoms) > 0) or (type(points) == Point and not points.is_empty):
        if type(points) == MultiPoint:
            points_list = [(p.x, p.y) for p in points.geoms]
            # print("List", points_list)
            point = random.choice(points_list)
        else:  # is Point
            point = (points.x, points.y)
        # print("Chosen point", point)
        chosen_points.append(point)
        circle = Point(point).buffer(2 * radius)
        points = points.difference(circle)
        # print(points)
    
    for x, y in chosen_points:
        sketch.circle(x, y, radius=1e-4)

    return sketch


def draw_dot_evenly_shaded_rect(x_0, y_0, width, height, density, dot_radius):
    sketch = get_empty_sketch() 
    sketch.translate(x_0, y_0)  
    sketch.rect(0, 0, width, height, mode="center")
    
    N_points = int(np.round(density * width * height))
    
    def sample_random_points_in_rect(N):
        x = np.random.uniform(-0.5 * width, 0.5 * width, size=N)
        y = np.random.uniform(-0.5 * height, 0.5 * height, size=N)
        return x, y
    
    sketch = draw_dot_evenly_shading(f_sample_points=sample_random_points_in_rect, sketch=sketch,
                                     N_points=N_points, radius=dot_radius)
    return sketch


def draw_dot_evenly_shaded_circle(x_0, y_0, radius, density, dot_radius):
    sketch = get_empty_sketch() 
    sketch.translate(x_0, y_0)  
    sketch.circle(0, 0, radius=radius)
    
    N_points = int(np.round(density * np.pi * radius**2))
    
    lambda_sample_point= lambda N: sample_random_points_on_circle(radius, N)
    sketch = draw_dot_evenly_shading(f_sample_points=lambda_sample_point, sketch=sketch,
                                     N_points=N_points, radius=dot_radius)
    return sketch


def rotate_and_draw_sketch(vsk, sketch, x=0, y=0, angle=0):
    with vsk.pushMatrix():
        vsk.translate(x, y)
        vsk.rotate(angle)
        vsk.sketch(sketch)
        