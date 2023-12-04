import numpy as np

def draw_line_thick(vsk, x_0, x_end, width, width_end=None, draw_lines=True, N_lines=10,
                    debug=False, debug_radius=0.05):
    if width_end is None: width_end = width
    diff = x_end - x_0
    angle = np.arctan2(diff[1], diff[0])
    s_angle = np.sin(angle)
    c_angle = np.cos(angle)
    vsk.polygon([x_0[0] + 0.5 * width * s_angle, x_end[0] + 0.5 * width_end * s_angle,
                    x_end[0] - 0.5 * width_end * s_angle, x_0[0] - 0.5 * width * s_angle],
                [x_0[1] - 0.5 * width * c_angle, x_end[1] - 0.5 * width_end * c_angle,
                    x_end[1] + 0.5 * width_end * c_angle, x_0[1] + 0.5 * width * c_angle], close=True)
    
    if draw_lines:
        for i in range(N_lines + 1):
            p = (i / (N_lines + 1)) * diff + x_0
            w = (i / (N_lines + 1)) * (width_end - width) + width
            vsk.line(p[0] - 0.5 * w * s_angle, p[1] + 0.5 * w * c_angle,
                     p[0] + 0.5 * w * s_angle, p[1] - 0.5 * w * c_angle)

    if debug:
        vsk.stroke(2)
        vsk.circle(x_0[0], x_0[1], radius=debug_radius)
        vsk.circle(x_end[0], x_end[1], radius=debug_radius)
        vsk.line(x_0[0], x_0[1], x_end[0], x_end[1])
        vsk.stroke(1)

def get_bezier_point(vsk, x_0, x_end, x_c1, x_c2=None, t=0):
    if x_c2 is None: x_c2 = x_c1.copy()
    point_x = vsk.bezierPoint(x_0[0], x_c1[0], x_c2[0], x_end[0], t)
    point_y = vsk.bezierPoint(x_0[1], x_c1[1], x_c2[1], x_end[1], t)
    return np.array([point_x, point_y])

def get_bezier_tangent(vsk, x_0, x_end, x_c1, x_c2=None, t=0):
    if x_c2 is None: x_c2 = x_c1.copy()
    x_tangent = vsk.bezierTangent(x_0[0], x_c1[0], x_c2[0], x_end[0], t)
    y_tangent = vsk.bezierTangent(x_0[1], x_c1[1], x_c2[1], x_end[1], t)
    return np.array([x_tangent, y_tangent])

def tangent_to_normal(x):
    normal = np.array([-x[1], x[0]])
    normal = normal / np.sqrt(normal[0]**2 + normal[1]**2)
    return normal

def get_bezier_normal(vsk, x_0, x_end, x_c1, x_c2=None, t=0):
    tangent = get_bezier_tangent(vsk, x_0, x_end, x_c1, x_c2, t)
    return tangent_to_normal(tangent)

def draw_bezier(vsk, x_0, x_end, x_c1, x_c2=None, debug=False, debug_radius=0.075):
    if debug:
        vsk.stroke(2)
        vsk.circle(x_0[0], x_0[1], radius=debug_radius)
        vsk.circle(x_c1[0], x_c1[1], radius=debug_radius)
        vsk.line(x_0[0], x_0[1], x_c1[0], x_c1[1])
        
        vsk.stroke(3)
        vsk.circle(x_end[0], x_end[1], radius=debug_radius)
        if x_c2 is None:
            vsk.line(x_end[0], x_end[1], x_c1[0], x_c1[1])
        else:
            vsk.circle(x_c2[0], x_c2[1], radius=debug_radius)
            vsk.line(x_end[0], x_end[1], x_c2[0], x_c2[1])
        vsk.stroke(1)
    if x_c2 is None: x_c2 = x_c1.copy()
    vsk.bezier(x_0[0], x_0[1], x_c1[0], x_c1[1], x_c2[0], x_c2[1], x_end[0], x_end[1])

def draw_bezier_thick(vsk, x_0, x_end, x_c1=None, x_c2=None, width=None, width_end=None,
                      N_segments=20, draw_lines=True, N_lines=10, debug=False, debug_radius=0.05):
    assert width is not None
    if width_end is None: width_end = width
    if x_c1 is None: x_c1 = 0.5 * (x_0 + x_end)
    
    points = np.zeros((2*N_segments,2))    
    for i in range(N_segments):
        t = i / (N_segments-1)
        width_i = np.interp(t, (0, 1), (width, width_end))
        point = get_bezier_point(vsk, x_0, x_end, x_c1, x_c2, t=t)
        normal = get_bezier_normal(vsk, x_0, x_end, x_c1, x_c2, t=t)
        # vsk.line(point[0], point[1], point[0] + normal[0], point[1] + normal[1])
        points[i] = point + 0.5 * width_i * normal
        points[2*N_segments-1-i] = point - 0.5 * width_i * normal

    vsk.polygon(points[:,0], points[:,1], close=True)
    
    if draw_lines:
        skips = int(N_segments / (N_lines))
        i_draw = int(np.round(0.5 * skips))
        for i in range(N_segments):
            if i % skips == i_draw:
                vsk.line(points[i,0], points[i,1], points[2*N_segments-1-i,0], points[2*N_segments-1-i,1])
    
    if debug:
        draw_bezier(vsk, x_0, x_end, x_c1, x_c2, debug=True, debug_radius=debug_radius)
