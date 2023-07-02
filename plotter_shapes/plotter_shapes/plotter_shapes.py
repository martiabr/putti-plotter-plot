import vsketch
import numpy as np
from numpy.random import default_rng
from enum import Enum

def pick_random_element(probs):
    return np.random.choice(len(probs), p=probs)


def enum_type_to_int(enum_type):
    return enum_type.value - 1


def get_empty_sketch(detail=1e-2):
    sketch = vsketch.Vsketch()
    sketch.detail(detail)
    return sketch


def draw_filled_circle(x, y, radius, line_width=1e-2):
    sketch = get_empty_sketch()
    N = int(radius / line_width)
    for r in np.linspace(radius, 0, N):
        sketch.circle(x, y, radius=r)
    return sketch


def draw_circle(x, y, radius):  
    sketch = get_empty_sketch()
    sketch.circle(x, y, radius=radius)
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


def sample_random_point_on_circle(radius):
    theta_i = np.random.uniform(0, 2 * np.pi)
    radius_i = radius * np.sqrt(np.random.uniform(0, 1.0))
    x_i, y_i = radius_i * np.array([np.cos(theta_i), np.sin(theta_i)])
    return x_i, y_i


def draw_speckled_shaded_circle(x_0, y_0, radius, density):
    sketch = get_empty_sketch()
    sketch.translate(x_0, y_0)
    sketch.circle(0, 0, radius=radius)
    
    N_speckles = int(np.round(density * np.pi * radius**2))
    for i in range(N_speckles):
        x_i, y_i = sample_random_point_on_circle(radius)
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


def rotate_and_draw_sketch(vsk, sketch, x, y, angle):
    with vsk.pushMatrix():
        vsk.translate(x, y)
        vsk.rotate(angle)
        vsk.sketch(sketch)
        