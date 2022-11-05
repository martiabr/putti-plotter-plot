from matplotlib.pyplot import fill
import numpy as np
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, std=1, lower=0, upper=10):
    return truncnorm(
        (lower - mean) / std, (upper - mean) / std, loc=mean, scale=std).rvs()
    
def draw_shaded_circle(vsk, x_0, y_0, radius, fill_distance, angle):
    vsk.circle(x_0, y_0, radius=radius)
    N = np.max((0, int(np.round(2 * (radius / fill_distance - 1)))))
    fill_distance = 2 * radius / (N + 1)
    with vsk.pushMatrix():
        vsk.translate(x_0, y_0)
        vsk.rotate(angle)
        for d in np.linspace(-radius + fill_distance, radius - fill_distance, N, endpoint=True):
            dy = radius * np.sin(np.arccos(d / radius))
            vsk.line(d, -dy, d, dy) 
