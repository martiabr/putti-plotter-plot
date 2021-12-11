from matplotlib import pyplot as plt
import numpy as np
import rasterio as rio
from scipy.ndimage import gaussian_filter, zoom

def trunc_map_lower(img, lower_bound):
    for x,y in np.ndindex(img.shape):
        if img[x, y] < lower_bound:
            img[x,y] = lower_bound
    return img
    
def smoothen_map(img, sigma):
    img = gaussian_filter(img, sigma * np.ones(2))
    return img

def crop_map(img, x_lim, y_lim):
    '''Crop the map image given by the upper and lower bounds in x and y.'''
    return img[y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]

def interp_map(img, scale):
    img_interp = zoom(img, scale)
    return img_interp

def show_map_image(img):
    plt.figure()
    heatmap = plt.imshow(img, cmap='hot', interpolation='nearest')
    plt.show()

def draw_contour_plot(img, levels, axis_off=True):
    plt.figure(figsize=(2*2.1,2*2.97))
    cs = plt.contour(img, levels=levels, colors='k', linewidths=0.25)
    plt.gca().invert_yaxis()
    plt.axis('equal')
    if axis_off: plt.gca().axis('off')
    return cs

def get_map_scale_factor(vsk, padding, width_img, width_paper=21):
    scale = (width_paper - 2 * padding) / width_img
    return scale

def draw_map(img, vsk, levels, axis_off=True, scale=1, offset=[0,0]):
    cs = draw_contour_plot(img, levels, axis_off)

    paths = cs.allsegs
    for i in range(len(paths)):
        for j in range(len(paths[i])):
            for k in range(len(paths[i][j])-1):
                vsk.line(scale * paths[i][j][k][0] + offset[0], scale * paths[i][j][k][1] + offset[1], scale * paths[i][j][k+1][0] + offset[0], scale * paths[i][j][k+1][1] + offset[1])
        
def draw_border(vsk, width=21, height=29.7, padding=0):
    vsk.line(padding, padding, width - padding, padding)
    vsk.line(padding, padding, padding, height - padding)
    vsk.line(width - padding, padding, width - padding, height - padding)
    vsk.line(padding, height - padding, width - padding, height - padding)

# Water drawing something?

def draw_water(vsk, img, period, water_level=0, radius=0.1, scale=1, offset=[0,0]):
    half_period = int(period / 2)
    for x,y in np.ndindex(img.shape):
        if x % period == half_period and y % period == half_period and img[x, y] < water_level:
            vsk.circle(scale*y + offset[0], scale*x + offset[1], radius)

def draw_title(vsk, title, font_size=20, padding=0, width=21, height=29.7):
    vsk.vpype(f"text -f rowmans -s {font_size} -p {width - padding}cm {height - padding + 0.025*font_size}cm --align right \"{title}\"")
    