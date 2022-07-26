import vsketch
import numpy as np
from enum import Enum
from curve_utils import draw_bezier, draw_line_thick, draw_bezier_thick
from plotter_util import get_truncated_normal

# - Should work such that I can call a series of functions and then draw any robot I want. 
#   Allows to create a drawing of every type possible, e.g. show every possible mouth, eyes, arms...
#   To have an overview.
# - Maybe have two scripts - robot editor, and random robot grid generator 

def draw_filled_circle(vsk, x, y, radius, line_width=0.02):
    N = int(radius/line_width)
    for r in np.linspace(radius, 0, N):
        vsk.circle(x, y, radius=r)

def draw_filled_rect(vsk, x, y, width, height, line_width=0.02):
    N = int(0.5 * np.min((width, height)) / line_width)
    for w, h in zip(np.linspace(width, width - 2 * N * line_width, N), np.linspace(height, height - 2 * N * line_width, N)):
        vsk.rect(x, y, w, h, mode="center")

def draw_striped_circle(vsk, x, y, radius, N, horizontal=True):
    vsk.circle(x, y, radius=radius)
    for p in np.linspace(-radius, radius, N+2):
        if horizontal:
            dx = radius * np.cos(np.arcsin(p / radius))
            vsk.line(x - dx, y + p, x + dx, y + p)
        else:
            dy = radius * np.sin(np.arccos(p / radius))
            vsk.line(x + p, y - dy, x + p, y + dy)


def test_bezier(vsk):
    x_start = np.array([0.0, 0.0])
    x_c1 = np.array([4.0, 0.0])
    x_c2 = np.array([2.0, 3.5])
    x_end = np.array([2.0, 5.0])
    draw_bezier_thick(vsk, x_start, x_end, x_c1, x_c2, width=1.0,
                      N_segments=25, N_lines=10, debug=False)
    
    x_start = np.array([-1.0, 0.0])
    x_c1 = np.array([-3.0, 1.0])
    x_c2 = np.array([-4.0, 2.0])
    x_end = np.array([-4.0, 6.0])
    draw_bezier_thick(vsk, x_start, x_end, x_c1, x_c2, width=0.75, width_end=1.5,
                      N_segments=25, N_lines=10, debug=False)


def test_line_thick(vsk):
    x_start = np.array([0.0, 0.0])
    x_end = np.array([2.0, 5.0])
    draw_line_thick(vsk, x_start, x_end, width=1.0, debug=True)
    
    x_start = np.array([-1.0, 0.0])
    x_end = np.array([-4.0, 6.0])
    draw_line_thick(vsk, x_start, x_end, width=0.75, width_end=1.5, N_lines=5, debug=True)
    
    x_start = np.array([-3.0, 0.0])
    x_end = np.array([-3.0, -3.0])
    draw_line_thick(vsk, x_start, x_end, width=0.5, debug=True)


def pick_random_element(elements, probs):
    return np.random.choice(len(elements), p=probs)

def enum_type_to_int(enum_type):
    return enum_type.value - 1

def generate_point_eye_sketch(detail="0.01"):
    eye_sketch = vsketch.Vsketch()
    eye_sketch.detail(detail)
    eye_sketch.circle(0, 0, 4e-2)
    return eye_sketch

def generate_ellipse_point_eye_sketch(width, frac=1.0, line_width=0.02, detail="0.01"):
    eye_sketch = vsketch.Vsketch()
    eye_sketch.fill(1)
    eye_sketch.detail(detail)
    N = int(width/line_width)
    for w in np.linspace(width, 0, N):
        eye_sketch.ellipse(0, 0, w, frac * w)
    eye_sketch.noFill()
    return eye_sketch

def generate_circle_eye_sketch(radius, x_pupil_gain=None, detail="0.01"):
    eye_sketch = vsketch.Vsketch()
    eye_sketch.detail(detail)
    eye_sketch.circle(0, 0, radius=radius)
    if x_pupil_gain is not None: eye_sketch.circle(x_pupil_gain * radius, 0, 4e-2)
    return eye_sketch

def generate_rect_eye_sketch(width, x_pupil_gain, detail="0.01"):
    eye_sketch = vsketch.Vsketch()
    eye_sketch.detail(detail)
    eye_sketch.rect(0, 0, width, width, mode="center")
    eye_sketch.circle(x_pupil_gain * width, 0, 4e-2)
    return eye_sketch

def generate_claw_sketch(base_width, claw_width, length_1, length_2, angle_1, angle_2, joint_radius=None, joint_bullet_length=False,
                         joint_point=False, use_pointy=False, detail="0.01"):
    hand_sketch = vsketch.Vsketch()
    hand_sketch.detail(detail)
    
    points = np.zeros((4, 2))
    points[0] = np.array([0, 0.5 * base_width])
    points[1] = points[0] + np.array([length_1 * np.cos(angle_1), length_1 * np.sin(angle_1)])
    points[2] = points[1] + np.array([claw_width * np.sin(angle_1), -claw_width * np.cos(angle_1)])
    points[-1] = points[0] + np.array([claw_width * np.sin(angle_1), -claw_width * np.cos(angle_1)])
    
    points_2 = np.zeros((3, 2)) if use_pointy else  np.zeros((4, 2))
    points_2[0] = points[1]
    points_2[1] = points_2[0] + np.array([length_2 * np.cos(angle_2), -length_2 * np.sin(angle_2)])
    if not use_pointy: points_2[2] = points_2[1] + np.array([-claw_width * np.sin(angle_2), -claw_width * np.cos(angle_2)])
    points_2[-1] = points_2[0] + np.array([-claw_width * np.sin(angle_2), -claw_width * np.cos(angle_2)])

    claw_shape = hand_sketch.createShape()
    claw_shape.rect(0.5 * claw_width, 0, claw_width, base_width, mode="center")
    claw_shape.polygon(points[:,0], points[:,1], close=True)
    claw_shape.polygon(points_2[:,0], points_2[:,1], close=True)
    claw_shape.polygon(points[:,0], -points[:,1], close=True)
    claw_shape.polygon(points_2[:,0], -points_2[:,1], close=True)
    
    hand_sketch.shape(claw_shape)
    
    if joint_radius is not None:
        if joint_bullet_length is not None:
            bullet_shape = hand_sketch.createShape()
            bullet_shape.arc(0, 0, 2*joint_radius, 2*joint_radius, -0.5*np.pi, 0.5*np.pi, close="chord", mode="center")  # Note: 1.01 factor is there for shapes to connect
            bullet_shape.rect(-0.4999 * joint_bullet_length, 0, joint_bullet_length, 2*joint_radius, mode="center")
            hand_sketch.shape(bullet_shape)
        else:
            hand_sketch.circle(0, 0, radius=joint_radius)
        if joint_point:
            hand_sketch.circle(0, 0, 1e-2)
    
    return hand_sketch

def generate_horse_shoe_sketch(width, upper_height, radius, detail="0.01"):
    hand_sketch = vsketch.Vsketch()
    hand_sketch.detail(detail)
    
    hand_shape = hand_sketch.createShape()
    hand_shape.arc(0, 0, 2*radius, 2*radius, 0.5 * np.pi, 1.5*np.pi, close="chord")
    hand_shape.rect(0.5*upper_height - 1e-6, 0, upper_height, 2*radius, mode="center")
    hand_shape.rect(0.5*upper_height - 1e-6, 0, upper_height, 2*(radius - width), mode="center", op="difference")
    hand_shape.arc(0, 0, 2*(radius - width), 2*(radius - width), 0.5 * np.pi, 1.5*np.pi, close="chord", op="difference")
    
    hand_sketch.translate(radius - 0.5 * width, 0)
    hand_sketch.shape(hand_shape)
    
    return hand_sketch

def pol2cart(radius, angle):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return np.array([x, y])
        
def generate_sawblade_sketch(N_blades, radius, flat_angle, rise_angle, lean_angle, blade_height_gain,
                             outer_ring_radius_gain=None, inner_ring_radius_gain=None, hole_ring_radius_gain=None,
                             sawblade_rod_width_gain=None, hole_points=False, rod_center_line=False, flipped_blade=False, detail="0.01"):
    hand_sketch = vsketch.Vsketch()
    hand_sketch.detail(detail)
    
    d_theta = 2 * np.pi / N_blades
    if flipped_blade: d_theta *= -1
    
    rod_length = radius * (1 + blade_height_gain + 0.4)
    hand_sketch.translate(rod_length, 0)
        
    points = np.zeros((N_blades * 5, 2))
    theta_end = 2 * np.pi
    if flipped_blade: theta_end *= -1
    for i, theta in enumerate(np.linspace(d_theta, theta_end, N_blades)):
        points[5*i] = pol2cart(radius, theta)
        points[5*i + 1] = pol2cart(radius, theta + flat_angle * d_theta)
        points[5*i + 2] = pol2cart(radius * (1 + blade_height_gain), theta + (flat_angle + rise_angle) * d_theta)
        points[5*i + 3] = pol2cart(radius * (1 + blade_height_gain), theta + (1.0 + lean_angle) * d_theta)
        points[5*i + 4] = pol2cart(radius, theta + d_theta)
    hand_sketch.polygon(points[:,0], points[:,1], close=True)
    
    # Additional details:
    if outer_ring_radius_gain is not None:
        hand_sketch.circle(0, 0, radius=outer_ring_radius_gain * radius)
    if inner_ring_radius_gain is not None:
        hand_sketch.circle(0, 0, radius=inner_ring_radius_gain * radius)
    
    if hole_ring_radius_gain is not None:
        hand_sketch.rect(-0.5 * rod_length, 0, rod_length, 2 * sawblade_rod_width_gain * hole_ring_radius_gain * radius, mode="center")    
        if rod_center_line: hand_sketch.line(0, 0, -rod_length, 0)    
        hand_sketch.circle(0, 0, radius=hole_ring_radius_gain * radius)
        if hole_points: hand_sketch.circle(0, 0, 4e-2)
        hand_sketch.circle(-rod_length, 0, radius=hole_ring_radius_gain * radius)
        if hole_points: hand_sketch.circle(-rod_length, 0, 4e-2)
    
    return hand_sketch

def generate_shoulder_sketch(width, height, detail="0.01"):
    shoulder_sketch = vsketch.Vsketch()
    shoulder_sketch.detail(detail)
    shoulder_sketch.rect(0.5 * width, 0, width, height, mode="center")  
    return shoulder_sketch

def generate_shoulder_circle_sketch(radius, detail="0.01"):
    shoulder_sketch = vsketch.Vsketch()
    shoulder_sketch.detail(detail)
    shoulder_sketch.circle(0, 0, radius=radius)  
    return shoulder_sketch

def generate_arm_stick_sketch(link_length_1, alpha_1, link_length_2, alpha_2, width, joint_radius,
                              joint_point=False, centerline=False, tube_line_width=None, shoulder_sketch=None, shoulder_width=None, hand_sketch=None,
                              detail="0.01", debug=False, debug_radius=0.05):
    arm_sketch = vsketch.Vsketch()
    arm_sketch.detail(detail)
    
    if tube_line_width is not None:
        N_1 = int(link_length_1 / tube_line_width)
        N_2 = int(link_length_2 / tube_line_width)
    
    with arm_sketch.pushMatrix():
        arm_sketch.rotate(alpha_1)
        with arm_sketch.pushMatrix():
            if shoulder_sketch is not None and shoulder_width is not None:
                arm_sketch.translate(shoulder_width, 0)
            
            if tube_line_width is not None and not centerline:
                draw_line_thick(arm_sketch, np.array([0, 0]), np.array([link_length_1, 0]), width, N_lines=N_1, debug=debug)
            else:
                arm_sketch.rect(0.5 * link_length_1, 0, link_length_1, width, mode="center")
            if centerline: arm_sketch.line(0, 0, link_length_1, 0)
            if debug:
                arm_sketch.stroke(2)
                arm_sketch.circle(0, 0, radius=debug_radius)
                arm_sketch.circle(link_length_1, 0, radius=debug_radius)
                arm_sketch.line(0, 0, link_length_1, 0)
                arm_sketch.stroke(1)
                
            arm_sketch.translate(link_length_1, 0)
            arm_sketch.rotate(alpha_2 - alpha_1)
            if tube_line_width is not None and not centerline:
                draw_line_thick(arm_sketch, np.array([0, 0]), np.array([link_length_2, 0]), width, N_lines=N_2, debug=debug)
            else:
                arm_sketch.rect(0.5 * link_length_2, 0, link_length_2, width, mode="center")
            if centerline: arm_sketch.line(0, 0, link_length_2, 0)
            arm_sketch.circle(0, 0, radius=joint_radius)
            if joint_point: arm_sketch.circle(0, 0, 1e-2)
            if debug:
                arm_sketch.stroke(2)
                arm_sketch.circle(0, 0, radius=debug_radius)
                arm_sketch.circle(link_length_2, 0, radius=debug_radius)
                arm_sketch.line(0, 0, link_length_2, 0)
                arm_sketch.stroke(1)
            
            if hand_sketch is not None:
                arm_sketch.translate(link_length_2, 0)
                arm_sketch.sketch(hand_sketch)
            
        if shoulder_sketch is not None and shoulder_width is not None:
            arm_sketch.sketch(shoulder_sketch)
        
    return arm_sketch


def generate_arm_tube_sketch(length, alpha, width, N_lines=20, shoulder_sketch=None, shoulder_width=None,
                             hand_sketch=None, detail="0.01", debug=False):
    arm_sketch = vsketch.Vsketch()
    arm_sketch.detail(detail)
    
    with arm_sketch.pushMatrix():
        arm_sketch.rotate(alpha)
        with arm_sketch.pushMatrix():
            if shoulder_sketch is not None and shoulder_width is not None:
                arm_sketch.translate(shoulder_width, 0)
            
            draw_line_thick(arm_sketch, np.array([0, 0]), np.array([length, 0]), width, N_lines=N_lines, debug=debug)
        
            if hand_sketch is not None:
                arm_sketch.translate(length, 0)
                arm_sketch.sketch(hand_sketch)
                
        if shoulder_sketch is not None and shoulder_width is not None:
            arm_sketch.sketch(shoulder_sketch)
            
    return arm_sketch
    

def generate_arm_tube_curve_sketch(x_end, y_end, x_c1, y_c1, x_c2, y_c2, width, width_end=None,
                        N_lines=20, shoulder_sketch=None, shoulder_width=None, hand_sketch=None, detail="0.01", debug=False):
    arm_sketch = vsketch.Vsketch()
    arm_sketch.detail(detail)
    
    with arm_sketch.pushMatrix():
        if shoulder_sketch is not None and shoulder_width is not None:
            arm_sketch.translate(shoulder_width, 0)
            
        draw_bezier_thick(arm_sketch, np.array([0.0, 0.0]), np.array([x_end, y_end]), np.array([x_c1, y_c1]),
                            np.array([x_c2, y_c2]), width=width, width_end=width_end, N_segments=40, N_lines=N_lines, debug=debug)
        
        if hand_sketch is not None:
            angle = np.arctan2(y_end - y_c2, x_end - x_c2)
            arm_sketch.translate(x_end, y_end)
            arm_sketch.rotate(angle)
            arm_sketch.sketch(hand_sketch)
    
    if shoulder_sketch is not None and shoulder_width is not None:
        arm_sketch.sketch(shoulder_sketch)
            
    return arm_sketch


def generate_leg_tube_sketch(length, width, N_lines=8, detail="0.01", debug=False):
    leg_sketch = vsketch.Vsketch()
    leg_sketch.detail(detail)
    draw_line_thick(leg_sketch, np.array([0.0, 0.0]), np.array([0, -length]), width=width, N_lines=N_lines, debug=debug)
    return leg_sketch

def generate_leg_omni_sketch(width, trapezoid_width, height, radius, radius_inner, N_shading=None, detail="0.01"):
    width_lower = trapezoid_width
    leg_sketch = vsketch.Vsketch()
    leg_sketch.detail(detail)
    with leg_sketch.pushMatrix():
        leg_sketch.translate(0, -radius)
        leg_sketch.circle(0, 0, radius=radius)
        if N_shading is not None:
            for angle in np.linspace(0, np.pi, N_shading):
                x, y = pol2cart(radius, angle)
                leg_sketch.line(x, y, 0, 0)
        leg_sketch.circle(0, 0, radius=radius_inner)
    leg_sketch.translate(0, -radius)
    leg_sketch.polygon([(0.5 * width_lower, 0), (0.5 * width, -height), (-0.5 * width, -height), (-0.5 * width_lower, 0)], close=True)
    return leg_sketch

def generate_leg_wheels_sketch(width, height, nut_height=None, nut_width=None, N_lines=0, nuts_both_sides=False, detail="0.01", debug=False):
    leg_sketch = vsketch.Vsketch()
    leg_sketch.detail(detail)
    
    draw_line_thick(leg_sketch, np.array([0.0, 0.0]), np.array([0, -height]), width=width, N_lines=N_lines, debug=debug)
    
    if nut_height is not None and nut_width is not None:
        leg_sketch.rect(0.5 * (width + nut_height), -0.5 * height, nut_height, nut_width, mode="center")
        if nuts_both_sides:
            leg_sketch.rect(-0.5 * (width + nut_height), -0.5 * height, nut_height, nut_width, mode="center")
    
    return leg_sketch


def generate_foot_arc_sketch(width, height, detail="0.01", debug=False):
    foot_sketch = vsketch.Vsketch()
    foot_sketch.detail(detail)
    foot_sketch.arc(0, 0, width, height, 0, np.pi, close="chord")
    return foot_sketch
    
def generate_foot_shoe_sketch(width, height, detail="0.01", debug=False):
    foot_sketch = vsketch.Vsketch()
    foot_sketch.detail(detail)
    
    hand_shape = foot_sketch.createShape()
    hand_shape.rect(0, -0.5 * height, width - height, height, mode="center")
    hand_shape.arc(0.5 * (width - height), 0, 2 * height, 2 * height, 0, np.pi, close="chord")
    foot_sketch.shape(hand_shape)
    return foot_sketch

def generate_antenna_sketch(base_width, base_height, antenna_width, antenna_height, antenna_radius, rect=True, detail="0.01"):
    antenna_sketch = vsketch.Vsketch()
    antenna_sketch.detail(detail)
    antenna_sketch.translate(0, -base_height)
    if rect:
        antenna_sketch.rect(0, 0.5 * base_height, base_width, base_height, mode="center")
    else:
        antenna_sketch.arc(0, base_height, base_width, 2*base_height, 0, np.pi, close="chord")
    antenna_sketch.translate(0, -0.5 * (antenna_height))
    antenna_sketch.rect(0, 0, antenna_width, antenna_height, mode="center")
    antenna_sketch.translate(0, -0.5 * (antenna_height + antenna_radius))
    antenna_sketch.circle(0, 0, antenna_radius)
    return antenna_sketch


### Bodies:

def draw_rect_body(vsk, width, height, x=0, y=0, radius=0, draw_inner_rect=False, inner_padding=0.0):
    vsk.rect(x, y, width, height, radius, mode="center")
    if draw_inner_rect:
        vsk.rect(x, y, width - inner_padding, height - inner_padding, radius, mode="center")
        
def draw_circle_body(vsk, radius, x=0, y=0, draw_inner_circle=False, inner_padding=0.0):
    vsk.circle(x, y, radius)
    if draw_inner_circle:
        vsk.circle(x, y, np.max(radius - inner_padding, 0))
    
def draw_bullet_body(vsk, width, arc_height, lower_height, x=0, y=0):
    bullet_body_shape = vsk.createShape()
    bullet_body_shape.arc(x, y, width, arc_height, 0, 1.01*np.pi, close="chord", mode="center")  # Note: 1.01 factor is there for shapes to connect
    bullet_body_shape.rect(x, y + 0.5 * lower_height, width, lower_height, mode="center")
    vsk.shape(bullet_body_shape)


### Neck:

def draw_neck(vsk, length, width, N_lines=8, debug=False):
    draw_line_thick(vsk, np.array([0.0, 0.0]), np.array([0, -length]), width=width, N_lines=N_lines, debug=debug)


### Heads:

def draw_trapezoid_head(vsk, width, upper_width_gain, height, x=0, y=0):
    upper_width = upper_width_gain * width
    with vsk.pushMatrix():
        vsk.translate(x, y)
        vsk.polygon([(0.5 * width, 0.5 * height), (0.5 * upper_width, -0.5 * height), (-0.5 * upper_width, -0.5 * height), (-0.5 * width, 0.5 * height)], close=True)
    
### Mouths:

def draw_smile_mouth(vsk, width, height, debug=False):
    draw_bezier(vsk, np.array([-0.5 * width, -0.5 * height]), np.array([0.5 * width, -0.5 * height]), np.array([-0.5 * width, 0.5 * height]), np.array([0.5 * width, 0.5 * height]), debug=debug)

def draw_grill_mouth(vsk, width, height, N_lines, rounded=True, debug=False):
    if rounded:
        mouth_shape = vsk.createShape()
        radius = 0.5 * height
        if width > 2 * radius:
            mouth_shape.rect(0, 0, width - 2 * radius, height, mode="center")
        mouth_shape.circle(0.5 * width - radius, 0, radius=radius)
        mouth_shape.circle(-0.5 * width + radius, 0, radius=radius)
        vsk.shape(mouth_shape)
        
        dx = width / (N_lines + 1)
        for x in np.linspace(dx, width - dx, N_lines):
            if x < radius:
                y = np.sqrt(radius**2 - (x - radius)**2)
            elif x > width - radius:
                y = np.sqrt(radius**2 - (x - width + radius)**2)
            else:
                y = 0.5 * height
            vsk.line(x - 0.5 * width, -y, x - 0.5 * width, y)
    else:
        draw_line_thick(vsk, np.array([-0.5 * width, 0]), np.array([0.5 * width, 0]), height, N_lines=N_lines, debug=debug)


class RobotsSketch(vsketch.SketchClass):
    # Main parameters:
    n_x = vsketch.Param(3, min_value=1)
    n_y = vsketch.Param(4, min_value=1)
    grid_dist_x = vsketch.Param(8.0)
    grid_dist_y = vsketch.Param(9.0)
    scale = vsketch.Param(0.7, decimals=2)
    
    debug = vsketch.Param(False)
    occult = vsketch.Param(True)
    
    # Body parameters:
    body_types = Enum('BodyType', 'RECT CIRCLE BULLET')
    body_rect_prob = vsketch.Param(0.5, min_value=0, max_value=1)
    body_circle_prob = vsketch.Param(0.0, min_value=0, max_value=1)
    body_bullet_prob = vsketch.Param(0.5, min_value=0, max_value=1)
    
    body_rect_width_min = vsketch.Param(1.5, min_value=0)
    body_rect_width_max = vsketch.Param(3.5, min_value=0)
    body_rect_height_min = vsketch.Param(1.5, min_value=0)
    body_rect_height_max = vsketch.Param(3.5, min_value=0)
    
    inner_rect_prob = vsketch.Param(0.3, min_value=0, max_value=1)
    body_rect_inner_padding_min = vsketch.Param(0.2)
    body_rect_inner_padding_max = vsketch.Param(0.75)
    
    body_circle_radius_min = vsketch.Param(1.2, min_value=0)
    body_circle_radius_max = vsketch.Param(2.5, min_value=0)
    
    body_inner_circle_prob = vsketch.Param(1.0, min_value=0, max_value=1)
    body_circle_inner_padding_min = vsketch.Param(0.2)
    body_circle_inner_padding_max = vsketch.Param(0.5)
    
    body_bullet_radius_min = vsketch.Param(0.75, min_value=0)
    body_bullet_radius_max = vsketch.Param(1.5, min_value=0)
    body_bullet_lower_height_min = vsketch.Param(0.5, min_value=0)
    body_bullet_lower_height_max = vsketch.Param(2.0, min_value=0)
    
    body_face_frame_prob = vsketch.Param(0.45, min_value=0, max_value=1)
    body_face_frame_pad_gain_min = vsketch.Param(0.04, min_value=0)
    body_face_frame_pad_gain_max = vsketch.Param(0.1, min_value=0)
    
    # Panel parameters:
    panel_prob = vsketch.Param(0.7, min_value=0, max_value=1)
    panel_double_prob = vsketch.Param(0.6, min_value=0, max_value=1)
    panel_two_doubles_prob = vsketch.Param(0.6, min_value=0, max_value=1)
    panel_outer_padding_gain_min = vsketch.Param(0.15, min_value=0)
    panel_outer_padding_gain_max = vsketch.Param(0.6, min_value=0)
    panel_inner_padding_gain_min = vsketch.Param(0.05, min_value=0)
    panel_inner_padding_gain_max = vsketch.Param(0.1, min_value=0)
    
    panel_frame_prob = vsketch.Param(0.9, min_value=0)
    panel_min_size = vsketch.Param(0.1, min_value=0)
    
    panel_single_frame_prob = vsketch.Param(0.3, min_value=0)
    panel_single_N_min = vsketch.Param(1, min_value=0)
    panel_single_N_max = vsketch.Param(3, min_value=0)
    
    panel_double_frame_prob = vsketch.Param(0.3, min_value=0)
    panel_double_N_min = vsketch.Param(1, min_value=0)
    panel_double_N_max = vsketch.Param(5, min_value=0)
    panel_double_pad_min = vsketch.Param(0.08, min_value=0)
    panel_double_pad_max = vsketch.Param(0.2, min_value=0)
    
    panel_double_types = Enum('DoublePanelType', 'NONE, CIRCLE RECT LINE')
    panel_double_none_prob = vsketch.Param(0.1, min_value=0)
    panel_double_circle_prob = vsketch.Param(0.3, min_value=0)
    panel_double_rect_prob = vsketch.Param(0.3, min_value=0)
    panel_double_line_prob = vsketch.Param(0.3, min_value=0)
    
    panel_double_circle_types = Enum('DoublePanelCircleType', 'SINGLE FILLED STRIPED DOT')
    panel_double_circle_single_prob = vsketch.Param(0.30, min_value=0)
    panel_double_circle_filled_prob = vsketch.Param(0.15, min_value=0)
    panel_double_circle_striped_prob = vsketch.Param(0.35, min_value=0)
    panel_double_circle_dot_prob = vsketch.Param(0.2, min_value=0)
    
    panel_double_circle_N_lines_min = vsketch.Param(1, min_value=0)
    panel_double_circle_N_lines_max = vsketch.Param(5, min_value=0)
    panel_double_circle_striped_horizontal_prob = vsketch.Param(0.5, min_value=0)
    panel_double_circle_radius_gain_std = vsketch.Param(0.4, min_value=0)
    panel_double_circle_radius_gain_min = vsketch.Param(0.4, min_value=0)

    panel_double_rect_types = Enum('DoublePanelRectType', 'SINGLE FILLED STRIPED')
    panel_double_rect_single_prob = vsketch.Param(0.40, min_value=0)
    panel_double_rect_filled_prob = vsketch.Param(0.10, min_value=0)
    panel_double_rect_striped_prob = vsketch.Param(0.50, min_value=0)

    panel_double_rect_N_lines_min = vsketch.Param(1, min_value=0)
    panel_double_rect_N_lines_max = vsketch.Param(5, min_value=0)

    panel_double_line_N_lines_min = vsketch.Param(1, min_value=0)
    panel_double_line_N_lines_max = vsketch.Param(5, min_value=0)
    # panel_double_line_pad_min = vsketch.Param(0.04, min_value=0)
    # panel_double_line_pad_max = vsketch.Param(0.1, min_value=0)
    
    # Head parameters:
    head_prob = vsketch.Param(0.5, min_value=0, max_value=1)
    head_types = Enum('HeadType', 'RECT BULLET TRAPEZOID')
    head_rect_prob = vsketch.Param(0.4, min_value=0, max_value=1)
    head_bullet_prob = vsketch.Param(0.3, min_value=0, max_value=1)
    head_trapezoid_prob = vsketch.Param(0.3, min_value=0, max_value=1)

    head_rect_width_min = vsketch.Param(0.9, min_value=0)
    head_rect_width_max = vsketch.Param(2.0, min_value=0)
    head_rect_height_min = vsketch.Param(0.9, min_value=0)
    head_rect_height_max = vsketch.Param(2.0, min_value=0)
    
    head_bullet_radius_min = vsketch.Param(0.6, min_value=0)
    head_bullet_radius_max = vsketch.Param(1.0, min_value=0)
    head_bullet_lower_height_min = vsketch.Param(0.25, min_value=0)
    head_bullet_lower_height_max = vsketch.Param(1.0, min_value=0)
    
    head_trapezoid_width_min = vsketch.Param(1.0, min_value=0)
    head_trapezoid_width_max = vsketch.Param(2.0, min_value=0)
    head_trapezoid_upper_width_gain_min = vsketch.Param(0.6, min_value=0)
    head_trapezoid_upper_width_gain_max = vsketch.Param(0.8, min_value=0)
    head_trapezoid_height_min = vsketch.Param(1.0, min_value=0)
    head_trapezoid_height_max = vsketch.Param(1.8, min_value=0)
    
    # Neck parameters:
    neck_prob = vsketch.Param(0.75, min_value=0, max_value=1)
    
    neck_width_min = vsketch.Param(0.2, min_value=0)
    neck_width_max = vsketch.Param(0.6, min_value=0)
    neck_length_min = vsketch.Param(0.2, min_value=0)
    neck_length_max = vsketch.Param(0.8, min_value=0)
    neck_N_lines_min =  vsketch.Param(0, min_value=0)
    neck_N_lines_max =  vsketch.Param(6, min_value=0)
    
    # Eye parameters:
    eye_types = Enum('EyeType', 'POINT ELLIPSE_POINT CIRCLE RECT SINGLE_CIRCLE CIRCLE_EMPTY')
    eye_point_prob = vsketch.Param(0.15, min_value=0, max_value=1)
    eye_ellipse_point_prob = vsketch.Param(0.1, min_value=0, max_value=1)
    eye_circle_prob = vsketch.Param(0.25, min_value=0, max_value=1)
    eye_rect_prob = vsketch.Param(0.1, min_value=0, max_value=1)
    eye_circle_single_prob = vsketch.Param(0.2, min_value=0, max_value=1)
    eye_circle_empty_prob = vsketch.Param(0.2, min_value=0, max_value=1)
    
    eye_circle_radius_gain_min = vsketch.Param(0.06, min_value=0)
    eye_circle_radius_gain_max = vsketch.Param(0.1, min_value=0)
    
    eye_rect_width_gain_min = vsketch.Param(0.08, min_value=0)
    eye_rect_width_gain_max = vsketch.Param(0.18, min_value=0)
    
    eye_single_circle_radius_gain_min = vsketch.Param(0.08, min_value=0)
    eye_single_circle_radius_gain_max = vsketch.Param(0.15, min_value=0)
    
    eye_circle_empty_radius_gain_min = vsketch.Param(0.02, min_value=0)
    eye_circle_empty_radius_gain_max = vsketch.Param(0.06, min_value=0)
    
    eye_circle_x_pupil_gain_max = vsketch.Param(0.2, min_value=0)
    
    eye_ellipse_width_min = vsketch.Param(0.02, min_value=0)
    eye_ellipse_width_max = vsketch.Param(0.06, min_value=0)
    eye_ellipse_frac_min = vsketch.Param(1.25, min_value=0)
    eye_ellipse_frac_max = vsketch.Param(2.0, min_value=0)
    
    eye_x_gain_min = vsketch.Param(0.1, min_value=0)
    eye_x_gain_max = vsketch.Param(0.6, min_value=0)
    eye_y_gain_max = vsketch.Param(0.1)
    
    # Mouth parameters:
    mouth_types = Enum('MouthType', 'NONE SMILE GRILL LINE')
    mouth_none_prob = vsketch.Param(0.1, min_value=0)
    mouth_smile_prob = vsketch.Param(0.25, min_value=0)
    mouth_grill_prob = vsketch.Param(0.4, min_value=0)
    mouth_line_prob = vsketch.Param(0.25, min_value=0)
    
    mouth_smile_width_gain_min = vsketch.Param(0.05, min_value=0)
    mouth_smile_width_gain_max = vsketch.Param(0.3, min_value=0)
    mouth_smile_height_gain_min = vsketch.Param(0.1, min_value=0)
    mouth_smile_height_gain_max = vsketch.Param(0.3, min_value=0)
    
    mouth_grill_width_gain_min = vsketch.Param(0.3, min_value=0)
    mouth_grill_width_gain_max = vsketch.Param(0.8, min_value=0)
    mouth_grill_height_gain_min = vsketch.Param(0.075, min_value=0)
    mouth_grill_height_gain_max = vsketch.Param(0.5, min_value=0)
    mouth_grill_N_lines_min = vsketch.Param(3, min_value=0)
    mouth_grill_N_lines_max = vsketch.Param(8, min_value=0)
    mouth_grill_rounded_prob = vsketch.Param(0.5, min_value=0)
    
    mouth_line_width_gain_min = vsketch.Param(0.3, min_value=0)
    mouth_line_width_gain_max = vsketch.Param(0.6, min_value=0)
    
    mouth_y_gain_min = vsketch.Param(0.05, min_value=0)
    mouth_y_gain_max = vsketch.Param(0.3, min_value=0)

    # Antenna parameters:
    antenna_prob = vsketch.Param(0.4, min_value=0)
    antenna_single_prob = vsketch.Param(0.4, min_value=0)
    antenna_rect_base_prob = vsketch.Param(0.5, min_value=0)
    antenna_base_width_min = vsketch.Param(0.2, min_value=0)
    antenna_base_width_max = vsketch.Param(0.3, min_value=0)
    antenna_base_height_min = vsketch.Param(0.1, min_value=0)
    antenna_base_height_max = vsketch.Param(0.2, min_value=0)
    antenna_width_min = vsketch.Param(0.04, min_value=0)
    antenna_width_max = vsketch.Param(0.08, min_value=0)
    antenna_height_min = vsketch.Param(0.2, min_value=0)
    antenna_height_max = vsketch.Param(1.0, min_value=0)
    antenna_radius_min = vsketch.Param(0.1, min_value=0)
    antenna_radius_max = vsketch.Param(0.3, min_value=0)
    antenna_x_gain_min = vsketch.Param(0.2, min_value=0)
    antenna_x_gain_max = vsketch.Param(0.4, min_value=0)
    
    # Arm parameters:
    arm_types = Enum('ArmType', 'NONE TUBE TUBE_CURVE STICK')
    arm_none_prob = vsketch.Param(0.0, min_value=0)
    arm_tube_prob = vsketch.Param(0.2, min_value=0)
    arm_tube_curve_prob = vsketch.Param(0.3, min_value=0)
    arm_stick_prob = vsketch.Param(0.5, min_value=0)
    
    arm_rect_y_gain_min = vsketch.Param(-0.3)
    arm_rect_y_gain_max = vsketch.Param(0.1)
    arm_bullet_y_gain_max = vsketch.Param(0.4, min_value=0)
    
    arm_tube_width_min = vsketch.Param(0.15, min_value=0)
    arm_tube_width_max = vsketch.Param(0.40, min_value=0)
    arm_tube_angle_min = vsketch.Param(-60)
    arm_tube_angle_max = vsketch.Param(60)
    arm_tube_length_min = vsketch.Param(0.5, min_value=0)
    arm_tube_length_max = vsketch.Param(1.5, min_value=0)
    arm_tube_N_lines_min =  vsketch.Param(5, min_value=0)
    arm_tube_N_lines_max =  vsketch.Param(20, min_value=0)
    
    arm_tube_curve_x_end_min = vsketch.Param(0.5, min_value=0)
    arm_tube_curve_x_end_max = vsketch.Param(1.0, min_value=0)
    arm_tube_curve_y_end_min = vsketch.Param(0.5, min_value=0)
    arm_tube_curve_y_end_max = vsketch.Param(1.0, min_value=0)
    arm_tube_curve_x_c1_min = vsketch.Param(0.3, min_value=0)
    arm_tube_curve_x_c1_max = vsketch.Param(0.7, min_value=0)
    arm_tube_curve_x_c2_min = vsketch.Param(0.3, min_value=0)
    arm_tube_curve_x_c2_max = vsketch.Param(0.7, min_value=0)
    arm_tube_curve_y_c2_min = vsketch.Param(-0.3, min_value=0)
    arm_tube_curve_y_c2_max = vsketch.Param(0.3, min_value=0)
    arm_tube_curve_up_prob = vsketch.Param(0.5, min_value=0)
    arm_tube_curve_flip_prob = vsketch.Param(0.2, min_value=0)
    
    arm_stick_width_min = vsketch.Param(0.06, min_value=0)
    arm_stick_width_max = vsketch.Param(0.25, min_value=0)
    arm_stick_length_1_min = vsketch.Param(0.4, min_value=0)
    arm_stick_length_1_max = vsketch.Param(0.8, min_value=0)
    arm_stick_length_2_min = vsketch.Param(0.4, min_value=0)
    arm_stick_length_2_max = vsketch.Param(0.8, min_value=0)
    arm_stick_angle_1_min = vsketch.Param(-60)
    arm_stick_angle_1_max = vsketch.Param(60)
    arm_stick_angle_2_min = vsketch.Param(-80)
    arm_stick_angle_2_max = vsketch.Param(80)
    arm_stick_joint_radius_min = vsketch.Param(0.075, min_value=0)
    arm_stick_joint_radius_max = vsketch.Param(0.15, min_value=0)
    arm_stick_joint_point_prob = vsketch.Param(0.3, min_value=0)
    arm_stick_centerline_prob = vsketch.Param(0.25, min_value=0)
    arm_stick_tube_prob = vsketch.Param(0.4, min_value=0)
    arm_stick_tube_line_width_min = vsketch.Param(0.02, min_value=0)
    arm_stick_tube_line_width_max = vsketch.Param(0.3, min_value=0)
    
    # Shoulder parameters:
    shoulder_types = Enum('ShoulderType', 'NONE RECT CIRCLE')
    shoulder_none_prob = vsketch.Param(0.3, min_value=0, max_value=1)
    shoulder_rect_prob = vsketch.Param(0.35, min_value=0, max_value=1)
    shoulder_circle_prob = vsketch.Param(0.35, min_value=0, max_value=1)
    arm_shoulder_width_min = vsketch.Param(0.20, min_value=0)
    arm_shoulder_width_max = vsketch.Param(0.5, min_value=0)
    arm_shoulder_width_to_arm_width_gain_min = vsketch.Param(1.3, min_value=0)
    arm_shoulder_height_min = vsketch.Param(0.20, min_value=0)
    arm_shoulder_height_max = vsketch.Param(0.6, min_value=0)
    arm_shoulder_radius_min = vsketch.Param(0.15, min_value=0)
    arm_shoulder_radius_max = vsketch.Param(0.3, min_value=0)
    arm_shoulder_angle_max = vsketch.Param(20, min_value=0)  # +- outside this deg angle we will not draw shoulders bause it will look weird
    shoulder_width_gain = vsketch.Param(0.8, min_value=0)
    
    # Hand parameters:
    hand_types = Enum('HandType', 'NONE CLAW HORSE_SHOE SAWBLADE')
    hand_none_prob = vsketch.Param(0.1, min_value=0)
    hand_claw_prob = vsketch.Param(0.5, min_value=0)
    hand_horse_shoe_prob = vsketch.Param(0.3, min_value=0)
    hand_sawblade_prob = vsketch.Param(0.1, min_value=0)
    
    hand_claw_length_1_min = vsketch.Param(0.3, min_value=0)
    hand_claw_length_1_max = vsketch.Param(0.6, min_value=0)
    hand_claw_length_2_min = vsketch.Param(0.3, min_value=0)
    hand_claw_length_2_max = vsketch.Param(0.4, min_value=0)
    hand_claw_angle_1_min = vsketch.Param(0.5, min_value=0)
    hand_claw_angle_1_max = vsketch.Param(1.0, min_value=0)
    hand_claw_angle_2_min = vsketch.Param(0.3, min_value=0)
    hand_claw_angle_2_max = vsketch.Param(0.7, min_value=0)
    hand_claw_width_min = vsketch.Param(0.1, min_value=0)
    hand_claw_width_max = vsketch.Param(0.2, min_value=0)
    hand_claw_joint_prob = vsketch.Param(0.5, min_value=0)
    hand_claw_bullet_prob = vsketch.Param(0.5, min_value=0)
    hand_claw_circle_point_prob = vsketch.Param(0.5, min_value=0)
    hand_claw_circle_radius_min = vsketch.Param(0.075, min_value=0)
    hand_claw_circle_radius_max = vsketch.Param(0.15, min_value=0)
    hand_claw_circle_radius_arm_width_gain_min = vsketch.Param(0.7, min_value=0)
    hand_claw_bullet_length_min = vsketch.Param(0.05, min_value=0)
    hand_claw_bullet_length_max = vsketch.Param(0.2, min_value=0)
    hand_claw_pointy_prob = vsketch.Param(0.3, min_value=0)

    hand_horse_shoe_width_min = vsketch.Param(0.1, min_value=0)
    hand_horse_shoe_width_max = vsketch.Param(0.3, min_value=0)
    hand_horse_shoe_upper_height_min = vsketch.Param(0.2, min_value=0)
    hand_horse_shoe_upper_height_max = vsketch.Param(0.3, min_value=0)
    hand_horse_shoe_radius_min = vsketch.Param(0.2, min_value=0)
    hand_horse_shoe_radius_max = vsketch.Param(0.35, min_value=0)
    hand_horse_shoe_radius_to_width_gain = vsketch.Param(0.7, min_value=0)

    hand_sawblade_N_blades_min = vsketch.Param(10, min_value=0)
    hand_sawblade_N_blades_max = vsketch.Param(15, min_value=0)
    hand_sawblade_flat_angle_min = vsketch.Param(0.2, min_value=0)
    hand_sawblade_flat_angle_max = vsketch.Param(0.3, min_value=0)
    hand_sawblade_rise_angle_min = vsketch.Param(0.4, min_value=0)
    hand_sawblade_rise_angle_max = vsketch.Param(0.6, min_value=0)
    hand_sawblade_lean_angle_min = vsketch.Param(0.2, min_value=0)
    hand_sawblade_lean_angle_max = vsketch.Param(0.4, min_value=0)
    hand_sawblade_radius_min = vsketch.Param(0.4, min_value=0)
    hand_sawblade_radius_max = vsketch.Param(0.8, min_value=0)
    hand_sawblade_height_gain_min = vsketch.Param(0.1, min_value=0)
    hand_sawblade_height_gain_max = vsketch.Param(0.25, min_value=0)
    hand_sawblade_outer_ring_prob = vsketch.Param(0.8, min_value=0)
    hand_sawblade_inner_ring_prob = vsketch.Param(0.8, min_value=0)
    hand_sawblade_outer_ring_radius_gain = vsketch.Param(0.9, min_value=0)
    hand_sawblade_inner_ring_radius_gain_min = vsketch.Param(0.3, min_value=0)
    hand_sawblade_inner_ring_radius_gain_max = vsketch.Param(0.7, min_value=0)
    hand_sawblade_hole_radius_gain_min = vsketch.Param(0.2, min_value=0)
    hand_sawblade_hole_radius_gain_max = vsketch.Param(0.3, min_value=0)
    hand_sawblade_rod_width_gain_min = vsketch.Param(0.5, min_value=0)
    hand_sawblade_rod_width_gain_max = vsketch.Param(1.0, min_value=0)
    hand_sawblade_hole_points_prob = vsketch.Param(0.6, min_value=0)
    hand_sawblade_rod_center_line_prob = vsketch.Param(0.6, min_value=0)
    
    # Leg parameters:
    leg_types = Enum('LegType', 'TUBE OMNI WHEELS WHEEL')
    leg_tube_prob = vsketch.Param(0.3, min_value=0)
    leg_omni_prob = vsketch.Param(0.2, min_value=0)
    leg_wheels_prob = vsketch.Param(0.3, min_value=0)
    leg_wheel_prob = vsketch.Param(0.2, min_value=0)
    
    leg_x_gain_max = vsketch.Param(0.40, min_value=0)
    leg_x_gain_min = vsketch.Param(0.15, min_value=0)
    
    leg_tube_width_max = vsketch.Param(0.40, min_value=0)
    leg_tube_width_min = vsketch.Param(0.1, min_value=0)
    leg_tube_length_min = vsketch.Param(0.4, min_value=0)
    leg_tube_length_max = vsketch.Param(1.75, min_value=0)
    leg_tube_N_lines_min =  vsketch.Param(0, min_value=0)
    leg_tube_N_lines_max =  vsketch.Param(20, min_value=0)
    
    leg_omni_width_gain_min = vsketch.Param(0.5, min_value=0)
    leg_omni_width_gain_max = vsketch.Param(0.9, min_value=0)
    leg_omni_trapezoid_width_gain_min = vsketch.Param(0.5, min_value=0)
    leg_omni_trapezoid_width_gain_max = vsketch.Param(0.7, min_value=0)
    leg_omni_height_gain_min = vsketch.Param(0.15, min_value=0)
    leg_omni_height_gain_max = vsketch.Param(0.4, min_value=0)
    leg_omni_radius_gain_min = vsketch.Param(0.1, min_value=0)
    leg_omni_radius_gain_max = vsketch.Param(0.5, min_value=0)
    leg_omni_inner_radius_min = vsketch.Param(0.05, min_value=0)
    leg_omni_inner_radius_max = vsketch.Param(0.1, min_value=0)
    leg_omni_shading_prob = vsketch.Param(0.3, min_value=0)
    leg_omni_N_shading_min = vsketch.Param(6, min_value=0)
    leg_omni_N_shading_max = vsketch.Param(12, min_value=0)

    leg_wheels_height_min = vsketch.Param(0.6, min_value=0)
    leg_wheels_height_max = vsketch.Param(1.2, min_value=0)
    leg_wheels_width_gain_min = vsketch.Param(0.4, min_value=0)
    leg_wheels_width_gain_max = vsketch.Param(0.8, min_value=0)
    leg_wheels_nut_height_min = vsketch.Param(0.1, min_value=0)
    leg_wheels_nut_height_max = vsketch.Param(0.2, min_value=0)
    leg_wheels_nut_width_gain_min = vsketch.Param(0.5, min_value=0)
    leg_wheels_nut_width_gain_max = vsketch.Param(0.9, min_value=0)
    leg_wheels_N_lines_min = vsketch.Param(5, min_value=0)
    leg_wheels_N_lines_max = vsketch.Param(10, min_value=0)
    
    leg_wheel_height_min = vsketch.Param(1.0, min_value=0)
    leg_wheel_height_max = vsketch.Param(2.0, min_value=0)
    leg_wheel_width_gain_min = vsketch.Param(0.4, min_value=0)
    leg_wheel_width_gain_max = vsketch.Param(0.6, min_value=0)
    leg_wheel_nut_height_min = vsketch.Param(0.2, min_value=0)
    leg_wheel_nut_height_max = vsketch.Param(0.5, min_value=0)
    leg_wheel_nut_width_gain_min = vsketch.Param(0.5, min_value=0)
    leg_wheel_nut_width_gain_max = vsketch.Param(0.9, min_value=0)
    leg_wheel_N_lines_min = vsketch.Param(5, min_value=0)
    leg_wheel_N_lines_max = vsketch.Param(15, min_value=0)
    
    # Foot parameters:
    foot_types = Enum('FootType', 'SHOE ARC')
    foot_shoe_prob = vsketch.Param(0.5, min_value=0)
    foot_arc_prob = vsketch.Param(0.5, min_value=0)
    
    foot_width_min = vsketch.Param(0.3, min_value=0)
    foot_width_max = vsketch.Param(0.8, min_value=0)
    
    foot_arc_height_gain_min = vsketch.Param(0.5, min_value=0)
    foot_arc_height_gain_max = vsketch.Param(1.0, min_value=0)
    
    foot_shoe_height_gain_min = vsketch.Param(0.3, min_value=0)
    foot_shoe_height_gain_max = vsketch.Param(0.5, min_value=0)
    
    
    class Node:
        def __init__(self, drawing, type, draw_type, x, y, width, height, layer, bottom=False):
            self.drawing = drawing
            self.type = type
            self.draw_type = draw_type
            self.x, self.y = x, y
            self.width, self.height = width, height
            self.layer = layer
            self.bottom = bottom
            self._children = []
        
        def add_child(self, node):
            self._children.append(node)
        
        def draw(self, vsk):
            if self.width > 0 and self.height > 0:
                if self.drawing.debug:
                    vsk.stroke(2)
                    vsk.rect(self.x, self.y, self.width, self.height, mode="center")
                    vsk.stroke(1)
                    
                if self.draw_type == "OUTER":
                    if np.random.random_sample() < self.drawing.panel_frame_prob:
                        vsk.rect(self.x, self.y, self.width, self.height, mode="center")
                # elif self.draw_type == "SINGLE":
                #     if self.drawing.debug:
                #         vsk.stroke(4)
                #         vsk.rect(self.x, self.y, self.width, self.height, mode="center")
                #         vsk.stroke(1)
                        
                #     if np.random.random_sample() < self.drawing.panel_single_frame_prob:
                #         vsk.rect(self.x, self.y, self.width, self.height, mode="center")
                    
                #     pad = np.random.uniform(self.drawing.panel_double_pad_min, self.drawing.panel_double_pad_max)
                        
                #     # vsk.circle(self.x, self.y, np.min((self.width, self.height)))
                elif self.draw_type == "SINGLE" or self.draw_type == "DOUBLE":
                    if self.draw_type == "SINGLE":
                        N = np.random.randint(self.drawing.panel_single_N_min, self.drawing.panel_single_N_max + 1)
                    elif self.draw_type == "DOUBLE":
                        N = np.random.randint(self.drawing.panel_double_N_min, self.drawing.panel_double_N_max + 1)
                    pad = np.random.uniform(self.drawing.panel_double_pad_min, self.drawing.panel_double_pad_max)
                    
                    if np.random.random_sample() < self.drawing.panel_double_frame_prob:
                        vsk.rect(self.x, self.y, self.width, self.height, mode="center")

                    if self.width > self.drawing.panel_min_size and self.height > self.drawing.panel_min_size:
                        horizontal = self.width > self.height
                        max_dim = np.max((self.width, self.height))
                        min_dim = np.min((self.width, self.height))
                        dp = max_dim / N
                        
                        for p in np.linspace(-0.5 * (max_dim - dp), 0.5 * (max_dim - dp), N):
                            panel_choice = pick_random_element(self.drawing.panel_double_types, self.drawing.panel_double_type_probs)
                            if panel_choice == enum_type_to_int(self.drawing.panel_double_types.CIRCLE):
                                panel_subchoice = pick_random_element(self.drawing.panel_double_circle_types, self.drawing.panel_double_circle_type_probs)
                            elif panel_choice == enum_type_to_int(self.drawing.panel_double_types.RECT):
                                panel_subchoice = pick_random_element(self.drawing.panel_double_rect_types, self.drawing.panel_double_rect_type_probs)
                            
                            if horizontal:
                                x_i = self.x + p
                                y_i = self.y
                            else:
                                x_i = self.x
                                y_i = self.y + p
                            
                            if self.drawing.debug:
                                if panel_choice == enum_type_to_int(self.drawing.panel_double_types.CIRCLE):
                                    vsk.stroke(5)
                                elif panel_choice == enum_type_to_int(self.drawing.panel_double_types.RECT):
                                    vsk.stroke(6)
                                elif panel_choice == enum_type_to_int(self.drawing.panel_double_types.LINE):
                                    vsk.stroke(7)
                                if horizontal:
                                    vsk.rect(x_i, y_i, dp, self.height, mode="center")
                                else: 
                                    vsk.rect(x_i, y_i, self.width, dp, mode="center")
                                vsk.stroke(1)
                
                            if panel_choice == enum_type_to_int(self.drawing.panel_double_types.CIRCLE):
                                radius_max = np.min((0.5 * (dp - pad), 0.5 * (min_dim - pad)))
                                radius_gain = get_truncated_normal(1.0, self.drawing.panel_double_circle_radius_gain_std, upper=1.0, lower=self.drawing.panel_double_circle_radius_gain_min)
                                radius = radius_gain * radius_max
                                if radius > 0:
                                    if panel_subchoice in (enum_type_to_int(self.drawing.panel_double_circle_types.SINGLE),
                                                        enum_type_to_int(self.drawing.panel_double_circle_types.DOT)):
                                        vsk.circle(x_i, y_i, radius=radius)
                                        if panel_subchoice == enum_type_to_int(self.drawing.panel_double_circle_types.DOT):
                                            vsk.circle(x_i, y_i, 4e-2)
                                    elif panel_subchoice == enum_type_to_int(self.drawing.panel_double_circle_types.FILLED):
                                        draw_filled_circle(vsk, x_i, y_i, radius)
                                    elif panel_subchoice == enum_type_to_int(self.drawing.panel_double_circle_types.STRIPED):
                                        N_lines = np.random.randint(self.drawing.panel_double_circle_N_lines_min, self.drawing.panel_double_circle_N_lines_max + 1)
                                        use_horizontal_stripes = np.random.random_sample() < self.drawing.panel_double_circle_striped_horizontal_prob
                                        draw_striped_circle(vsk, x_i, y_i, radius, N=N_lines, horizontal=use_horizontal_stripes)
                            elif panel_choice == enum_type_to_int(self.drawing.panel_double_types.RECT):
                                if horizontal:
                                    rect_width = dp - pad
                                    rect_height = self.height - pad
                                else:
                                    rect_height = dp - pad
                                    rect_width = self.width - pad  # TODO: randomize more with trunc normal
                                if rect_height > 0 and rect_width > 0:
                                    if panel_subchoice == enum_type_to_int(self.drawing.panel_double_rect_types.SINGLE):
                                        vsk.rect(x_i, y_i, rect_width, rect_height, mode="center")
                                    elif panel_subchoice == enum_type_to_int(self.drawing.panel_double_rect_types.FILLED):
                                        draw_filled_rect(vsk, x_i, y_i, rect_width, rect_height)
                                    elif panel_subchoice == enum_type_to_int(self.drawing.panel_double_rect_types.STRIPED):
                                        N_lines = np.random.randint(self.drawing.panel_double_rect_N_lines_min, self.drawing.panel_double_rect_N_lines_max + 1)
                                        if horizontal:
                                            draw_line_thick(vsk, np.array([x_i, y_i - 0.5 * rect_height]),
                                                            np.array([x_i, y_i + 0.5 * rect_height]), rect_width, N_lines=N_lines)
                                        else:
                                            draw_line_thick(vsk, np.array([x_i - 0.5 * rect_width, y_i]),
                                                            np.array([x_i + 0.5 * rect_width, y_i]), rect_height, N_lines=N_lines)
                            elif panel_choice == enum_type_to_int(self.drawing.panel_double_types.LINE):
                                horizontal_lines = (horizontal and (dp - pad) > (self.height - pad)) or (not horizontal and (dp - pad) < (self.width - pad))
                                N_lines = np.random.randint(self.drawing.panel_double_line_N_lines_min, self.drawing.panel_double_line_N_lines_max + 1)
                                # line_pad = np.random.uniform(self.drawing.panel_double_line_pad_min, self.drawing.panel_double_line_pad_max)
                                if horizontal_lines:  # TODO: this whole ifelse structure is horrible
                                    if horizontal:
                                        for r in np.linspace(-0.5 * (min_dim - pad), 0.5 * (min_dim - pad), N_lines):
                                            if N_lines == 1: r = 0
                                            vsk.line(x_i - 0.5 * (dp - pad), y_i + r, x_i + 0.5 * (dp - pad), y_i + r)
                                    else:
                                        for r in np.linspace(-0.5 * (dp - pad), 0.5 * (dp - pad), N_lines):
                                            if N_lines == 1: r = 0
                                            vsk.line(x_i - 0.5 * (self.width - pad), y_i + r, x_i + 0.5 * (self.width - pad), y_i + r)
                                else:
                                    if horizontal:
                                        for r in np.linspace(-0.5 * (dp - pad), 0.5 * (dp - pad), N_lines):
                                            if N_lines == 1: r = 0
                                            vsk.line(x_i + r, y_i - 0.5 * (self.height - pad), x_i + r, y_i + 0.5 * (self.height - pad))
                                    else:
                                        for r in np.linspace(-0.5 * (min_dim - pad), 0.5 * (min_dim - pad), N_lines):
                                            if N_lines == 1: r = 0
                                            vsk.line(x_i + r, y_i - 0.5 * (dp - pad), x_i + r, y_i + 0.5 * (dp - pad))
                                
        def __iter__(self):
            return iter(self._children)

        def breadth_first(self):
            q = [self]
            while q:
                n = q.pop(0)
                yield n
                for c in n._children:
                    q.append(c)


    class QuadTree:
        def __init__(self, drawing, width, height, outer_padding, inner_padding, x=0.0, y=0.0, layers=2):
            self.drawing = drawing
            self.outer_padding = outer_padding
            self.inner_padding = inner_padding
            self.width = width
            self.height = height
            self.x = x
            self.y = y
            self.layers = layers
            
            self.root = self.generate_tree(self.layers)
        
        def generate_children(self, node, layers):
            picks = []
            
            pick_double = np.random.random_sample() < self.drawing.panel_double_prob
            if pick_double:
                double_sections = [[-1,0], [1,0], [0,-1], [0,1]]
                pick = double_sections[np.random.randint(len(double_sections))]
                x, y = pick
                picks.append(pick)
                pick_two_doubles = pick_double and np.random.random_sample() < self.drawing.panel_two_doubles_prob
                if pick_two_doubles:
                    picks.append([-x, -y])
                else:
                    if x != 0:
                        picks.append([-x, 1])
                        picks.append([-x, -1])
                    else:
                        picks.append([1, -y])
                        picks.append([-1, -y]) 
            else:
                picks += [[-1,-1], [-1,1], [1,-1], [1,1]]
                
            for pick in picks:
                is_double = (pick[0] == 0) or (pick[1] == 0)
                x = 0.25 * node.width * pick[0]
                y = 0.25 * node.height * pick[1]
                
                if pick[0] == 0:
                    child_width = node.width - 2 * self.inner_padding
                else:
                    child_width = 0.5 * node.width - 2 * self.inner_padding
                if pick[1] == 0:
                    child_height = node.height - 2 * self.inner_padding
                else:
                    child_height = 0.5 * node.height - 2 * self.inner_padding
                
                if child_height > 1e-2 and child_width > 1e-2:
                    stop_prob = 0.5
                    do_stop = (np.random.random_sample() < stop_prob) or (node.layer == layers - 1)
                    bottom = do_stop or is_double
                    
                    if is_double:
                        draw_type="DOUBLE"
                    elif do_stop:
                        draw_type="SINGLE"
                    else:
                        draw_type="OUTER"
                        
                        
                    child = self.drawing.Node(self.drawing, "...", draw_type, x=node.x+x, y=node.y+y, width=child_width,
                                height=child_height, layer=node.layer+1, bottom=bottom)
                    if not bottom:
                        self.generate_children(child, layers)
                    node.add_child(child)
        
        
        def generate_tree(self, layers):
            inner_width = self.width - self.outer_padding
            inner_height = self.height - self.outer_padding
            
            root = self.drawing.Node(self.drawing, "root", "OUTER", x=0, y=0, width=inner_width, height=inner_height, layer=0)
            
            self.generate_children(root, layers)
            
            return root
        
        
        def draw(self, vsk):
            with vsk.pushMatrix():
                vsk.translate(self.x, self.y)
                for node in self.root.breadth_first():
                    node.draw(vsk)
            
    
    def draw_eyes(self, vsk, face_width, face_height):
        self.eye_choice = pick_random_element(self.eye_types, self.eye_type_probs)
        self.eye_radius = 0.0
        
        if self.eye_choice == enum_type_to_int(self.eye_types.POINT):
            eye_sketch = generate_point_eye_sketch()
        elif self.eye_choice == enum_type_to_int(self.eye_types.ELLIPSE_POINT):
            self.eye_radius = np.random.uniform(self.eye_ellipse_width_min, self.eye_ellipse_width_max)
            eye_frac = np.random.uniform(self.eye_ellipse_frac_min, self.eye_ellipse_frac_max)
            eye_sketch = generate_ellipse_point_eye_sketch(self.eye_radius, eye_frac)
        elif self.eye_choice == enum_type_to_int(self.eye_types.CIRCLE):
            self.eye_radius = np.random.uniform(self.eye_circle_radius_gain_min, self.eye_circle_radius_gain_max) * self.body_width
            pupil_x_gain = np.random.uniform(-self.eye_circle_x_pupil_gain_max, self.eye_circle_x_pupil_gain_max)
            eye_sketch = generate_circle_eye_sketch(self.eye_radius, pupil_x_gain)
        elif self.eye_choice == enum_type_to_int(self.eye_types.RECT):
            self.eye_radius = 0.5 * np.random.uniform(self.eye_rect_width_gain_min, self.eye_rect_width_gain_max) * self.body_width
            pupil_x_gain = np.random.uniform(-self.eye_circle_x_pupil_gain_max, self.eye_circle_x_pupil_gain_max)
            eye_sketch = generate_rect_eye_sketch(2 * self.eye_radius, pupil_x_gain)
        elif self.eye_choice == enum_type_to_int(self.eye_types.SINGLE_CIRCLE):
            self.eye_radius = np.random.uniform(self.eye_single_circle_radius_gain_min, self.eye_single_circle_radius_gain_max) * self.body_width
            pupil_x_gain = np.random.uniform(-self.eye_circle_x_pupil_gain_max, self.eye_circle_x_pupil_gain_max)
            eye_sketch = generate_circle_eye_sketch(self.eye_radius, pupil_x_gain)
        elif self.eye_choice == enum_type_to_int(self.eye_types.CIRCLE_EMPTY):
            self.eye_radius = np.random.uniform(self.eye_circle_empty_radius_gain_min, self.eye_circle_empty_radius_gain_max) * self.body_width
            eye_sketch = generate_ellipse_point_eye_sketch(self.eye_radius)
            
        self.eye_y = np.random.uniform(0.0, self.eye_y_gain_max) * face_height
        self.eye_x = 0.0
        if self.eye_choice == enum_type_to_int(self.eye_types.SINGLE_CIRCLE):
            with vsk.pushMatrix():
                vsk.translate(0, -self.eye_y)
                vsk.sketch(eye_sketch)
        else:
            eye_x_gain = np.random.uniform(self.eye_x_gain_min, self.eye_x_gain_max)
            self.eye_x = np.max((eye_x_gain * 0.5 * face_width, 1.2 * self.eye_radius))
            with vsk.pushMatrix():
                vsk.translate(self.eye_x, -self.eye_y)
                vsk.sketch(eye_sketch)
                vsk.translate(-2 * self.eye_x, 0)
                if np.random.random_sample() < 0.5: vsk.scale(-1, 1)
                vsk.sketch(eye_sketch)
                
                
    def draw_mouth(self, vsk, face_width, face_lower_height, debug=False):
        mouth_choice = pick_random_element(self.mouth_types, self.mouth_type_probs)
        with vsk.pushMatrix():
            
            self.mouth_width, self.mouth_height = 0, 0
            if mouth_choice == enum_type_to_int(self.mouth_types.SMILE):
                self.mouth_width = np.random.uniform(self.mouth_smile_width_gain_min, self.mouth_smile_width_gain_max) * face_width
                self.mouth_height = np.random.uniform(self.mouth_smile_height_gain_min, self.mouth_smile_height_gain_max) * face_lower_height
            elif mouth_choice == enum_type_to_int(self.mouth_types.GRILL):
                self.mouth_width = np.random.uniform(self.mouth_grill_width_gain_min, self.mouth_grill_width_gain_max) * face_width
                self.mouth_height = np.random.uniform(self.mouth_grill_height_gain_min, self.mouth_grill_height_gain_max) * face_lower_height
                mouth_N_lines = np.random.randint(self.mouth_grill_N_lines_min, self.mouth_grill_N_lines_max + 1)
                mouth_rounded = np.random.random_sample() < self.mouth_grill_rounded_prob
            elif mouth_choice == enum_type_to_int(self.mouth_types.LINE):
                self.mouth_width = np.random.uniform(self.mouth_line_width_gain_min, self.mouth_line_width_gain_max) * face_width
            
            y_gain = np.random.uniform(self.mouth_y_gain_min, self.mouth_y_gain_max)
            self.mouth_y_offset = y_gain * face_lower_height
            vsk.translate(0, self.mouth_y_offset + self.eye_radius + 0.5 * self.mouth_height)
            
            if mouth_choice == enum_type_to_int(self.mouth_types.SMILE):
                draw_smile_mouth(vsk, self.mouth_width, self.mouth_height, debug=debug)
            elif mouth_choice == enum_type_to_int(self.mouth_types.GRILL):
                draw_grill_mouth(vsk, self.mouth_width, self.mouth_height, mouth_N_lines, rounded=mouth_rounded, debug=debug)
            elif mouth_choice == enum_type_to_int(self.mouth_types.LINE):
                vsk.line(-0.5 * self.mouth_width, 0, 0.5 * self.mouth_width, 0)

    
    def draw_body(self, vsk, draw_face=False, debug=False):
        if self.body_choice == enum_type_to_int(self.body_types.RECT):
            use_inner_rect = np.random.random_sample() < self.inner_rect_prob
            if use_inner_rect:
                body_inner_padding = np.random.uniform(self.body_rect_inner_padding_min, self.body_rect_inner_padding_max)
            else:
                body_inner_padding = None
                
            draw_rect_body(vsk, self.body_width, self.body_height, draw_inner_rect=use_inner_rect, inner_padding=body_inner_padding)
        elif self.body_choice == enum_type_to_int(self.body_types.CIRCLE):
            use_inner_circle = np.random.random_sample() < self.body_inner_circle_prob
            if use_inner_circle:
                body_inner_padding = np.random.uniform(self.body_circle_inner_padding_min, self.body_circle_inner_padding_max)
            else:
                body_inner_padding = None
                
            draw_circle_body(vsk, self.body_width, draw_inner_circle=use_inner_circle, inner_padding=body_inner_padding)
        elif self.body_choice == enum_type_to_int(self.body_types.BULLET):
            draw_bullet_body(vsk, self.body_width, self.body_width, self.body_lower_height)
            
        if draw_face:
            self.draw_eyes(vsk, self.body_width, self.body_height)
            self.draw_mouth(vsk, self.body_width, self.body_lower_height, debug=debug)
            if np.random.random_sample() < self.body_face_frame_prob and self.body_choice == enum_type_to_int(self.body_types.RECT) and not use_inner_rect:
                pad = np.random.uniform(self.body_face_frame_pad_gain_min, self.body_face_frame_pad_gain_max) * np.min((self.body_width, self.body_height))
                width = np.max((self.mouth_width, 2 * (self.eye_x + self.eye_radius)))
                vsk.line(-0.5 * width - pad, -self.eye_y - self.eye_radius - pad, 0.5 * width + pad, -self.eye_y - self.eye_radius - pad)
                vsk.line(-0.5 * width - pad, self.mouth_y_offset + self.mouth_height + self.eye_radius + pad, 0.5 * width + pad, self.mouth_y_offset + self.mouth_height + self.eye_radius + pad)
                vsk.line(-0.5 * width - pad, -self.eye_y - self.eye_radius - pad, -0.5 * width - pad, self.mouth_y_offset + self.mouth_height + self.eye_radius + pad)
                vsk.line(0.5 * width + pad, -self.eye_y - self.eye_radius - pad, 0.5 * width + pad, self.mouth_y_offset + self.mouth_height + self.eye_radius + pad)
        else:
            if np.random.random_sample() < self.panel_prob:
                panel_outer_padding = np.random.uniform(self.panel_outer_padding_gain_min, self.panel_outer_padding_gain_max) * self.body_width
                panel_inner_padding = np.random.uniform(self.panel_inner_padding_gain_min, self.panel_inner_padding_gain_max) * self.body_width
                if self.body_choice == enum_type_to_int(self.body_types.RECT):
                    tree = self.QuadTree(self, self.body_width, self.body_height, panel_outer_padding, panel_inner_padding)
                    tree.draw(vsk)
                elif self.body_choice == enum_type_to_int(self.body_types.BULLET):
                    tree = self.QuadTree(self, self.body_width, self.body_lower_height, panel_outer_padding,
                                         panel_inner_padding, y=0.5*self.body_lower_height)
                    tree.draw(vsk)

    def draw_robot(self, vsk, debug=False):
        self.body_choice = pick_random_element(self.body_types, self.body_type_probs)
        if self.body_choice == enum_type_to_int(self.body_types.RECT):
            self.body_width = np.random.uniform(self.body_rect_width_min, self.body_rect_width_max)
            self.body_height = np.random.uniform(self.body_rect_height_min, self.body_rect_height_max)
            self.body_lower_height = 0.5 * self.body_height
            self.body_upper_height = 0.5 * self.body_height
        elif self.body_choice == enum_type_to_int(self.body_types.CIRCLE):
            self.body_width = np.random.uniform(self.body_circle_radius_min, self.body_circle_radius_max)
            self.body_height = self.body_width
            self.body_lower_height = 0.5 * self.body_height
            self.body_upper_height = 0.5 * self.body_height
        elif self.body_choice == enum_type_to_int(self.body_types.BULLET):
            self.body_width = 2 * np.random.uniform(self.body_bullet_radius_min, self.body_bullet_radius_max)
            # arc_height = np.random.uniform(self.body_bullet_radius_min, self.body_bullet_radius_max)
            self.body_lower_height = np.random.uniform(self.body_bullet_lower_height_min, self.body_bullet_lower_height_max)
            # self.body_upper_height = arc_height
            self.body_upper_height = 0.5 * self.body_width
            self.body_height = self.body_lower_height + self.body_upper_height
        
        # Legs: 
        leg_choice = pick_random_element(self.leg_types, self.leg_type_probs)
        
        if leg_choice == enum_type_to_int(self.leg_types.TUBE):  # two
            leg_x_gain = np.random.uniform(self.leg_x_gain_min, self.leg_x_gain_max)
        elif leg_choice == enum_type_to_int(self.leg_types.WHEELS): 
            leg_x_gain = 0.5
        
        if leg_choice == enum_type_to_int(self.leg_types.TUBE):
            leg_length = np.random.uniform(self.leg_tube_length_min, self.leg_tube_length_max)
            leg_width = np.random.uniform(self.leg_tube_width_min, self.leg_tube_width_max)
            leg_N_lines = np.random.randint(self.leg_tube_N_lines_min, self.leg_tube_N_lines_max + 1)
            leg_sketch = generate_leg_tube_sketch(leg_length, leg_width, N_lines=leg_N_lines, debug=debug)
        elif leg_choice == enum_type_to_int(self.leg_types.OMNI):
            leg_omni_width = np.random.uniform(self.leg_omni_width_gain_min, self.leg_omni_width_gain_max) * self.body_width
            leg_omni_trapezoid_width = np.random.uniform(self.leg_omni_trapezoid_width_gain_min, self.leg_omni_trapezoid_width_gain_max) * leg_omni_width
            leg_omni_height = np.random.uniform(self.leg_omni_height_gain_min, self.leg_omni_height_gain_max) * leg_omni_width
            leg_omni_radius = np.random.uniform(self.leg_omni_radius_gain_min, self.leg_omni_radius_gain_max) * leg_omni_trapezoid_width
            leg_omni_inner_radius = np.random.uniform(self.leg_omni_inner_radius_min, self.leg_omni_inner_radius_max)
            leg_omni_do_shading = np.random.random_sample() < self.leg_omni_shading_prob
            N_leg_omni_shading = None
            if leg_omni_do_shading: N_leg_omni_shading = np.random.randint(self.leg_omni_N_shading_min, self.leg_omni_N_shading_max + 1)
            leg_length = leg_omni_radius + leg_omni_height
            leg_sketch = generate_leg_omni_sketch(leg_omni_width, leg_omni_trapezoid_width, leg_omni_height, leg_omni_radius, leg_omni_inner_radius, N_shading=N_leg_omni_shading)
        elif leg_choice == enum_type_to_int(self.leg_types.WHEELS):
            leg_wheels_height = np.random.uniform(self.leg_wheels_height_min, self.leg_wheels_height_max)
            leg_wheels_width = np.random.uniform(self.leg_wheels_width_gain_min, self.leg_wheels_width_gain_max) * leg_wheels_height
            leg_wheels_nut_height = np.random.uniform(self.leg_wheels_nut_height_min, self.leg_wheels_nut_height_max)
            leg_wheels_nut_width = np.random.uniform(self.leg_wheels_nut_width_gain_min, self.leg_wheels_nut_width_gain_max) * leg_wheels_nut_height
            leg_wheels_N_lines = np.random.randint(self.leg_wheels_N_lines_min, self.leg_wheels_N_lines_max + 1)
            leg_length = 0.5 * leg_wheels_height
            leg_sketch = generate_leg_wheels_sketch(leg_wheels_width, leg_wheels_height, leg_wheels_nut_width, leg_wheels_nut_height, leg_wheels_N_lines)
        elif leg_choice == enum_type_to_int(self.leg_types.WHEEL):
            leg_wheel_height = np.random.uniform(self.leg_wheel_height_min, self.leg_wheel_height_max)
            leg_wheel_width = np.random.uniform(self.leg_wheel_width_gain_min, self.leg_wheel_width_gain_max) * leg_wheel_height
            leg_wheel_nut_height = np.random.uniform(self.leg_wheel_nut_height_min, self.leg_wheel_nut_height_max)
            leg_wheel_nut_width = np.random.uniform(self.leg_wheel_nut_width_gain_min, self.leg_wheel_nut_width_gain_max) * leg_wheel_nut_height
            leg_wheel_N_lines = np.random.randint(self.leg_wheel_N_lines_min, self.leg_wheel_N_lines_max + 1)
            leg_length = 0.5 * leg_wheel_height
            leg_sketch = generate_leg_wheels_sketch(leg_wheel_width, leg_wheel_height, leg_wheel_nut_width, leg_wheel_nut_height,
                                                    leg_wheel_N_lines, nuts_both_sides=True)
        
        # Foot:
        if leg_choice == enum_type_to_int(self.leg_types.TUBE):
            foot_choice = pick_random_element(self.foot_types, self.foot_type_probs)
            foot_width = np.max((np.random.uniform(self.foot_width_min, self.foot_width_max), 1.1 * leg_width))
            if foot_choice == enum_type_to_int(self.foot_types.ARC):
                foot_height = np.random.uniform(self.foot_arc_height_gain_min, self.foot_arc_height_gain_max) * foot_width
                foot_sketch = generate_foot_arc_sketch(foot_width, foot_height)
            elif foot_choice == enum_type_to_int(self.foot_types.SHOE):
                foot_height = np.random.uniform(self.foot_shoe_height_gain_min, self.foot_shoe_height_gain_max) * foot_width
                if foot_width - foot_height < 1.1 * leg_width:
                    foot_width = 1.1 * leg_width + foot_height
                foot_sketch = generate_foot_shoe_sketch(foot_width, foot_height)
            leg_sketch.sketch(foot_sketch)
                
        with vsk.pushMatrix():
            if leg_choice in (enum_type_to_int(self.leg_types.TUBE), enum_type_to_int(self.leg_types.WHEELS)):  # two sketches
                vsk.translate(leg_x_gain * self.body_width, 0)
                vsk.sketch(leg_sketch)
                vsk.translate(-2 * leg_x_gain * self.body_width, 0)
                vsk.scale(-1, 1)
                vsk.sketch(leg_sketch)
            else:
                vsk.sketch(leg_sketch)
        vsk.translate(0, -self.body_lower_height - leg_length)
        
               
        # Arms:
        arm_choice = pick_random_element(self.arm_types, self.arm_type_probs)
        if arm_choice != enum_type_to_int(self.arm_types.NONE):            
            arm_y = 0.0
            if self.body_choice == enum_type_to_int(self.body_types.BULLET):
                arm_y = self.body_lower_height * np.random.uniform(0, self.arm_bullet_y_gain_max)
            elif self.body_choice == enum_type_to_int(self.body_types.RECT):
                arm_y = self.body_height * np.random.uniform(self.arm_rect_y_gain_min, self.arm_rect_y_gain_max)
                
            if arm_choice == enum_type_to_int(self.arm_types.TUBE) or arm_choice == enum_type_to_int(self.arm_types.TUBE_CURVE):
                arm_width = np.random.uniform(self.arm_tube_width_min, self.arm_tube_width_max)
                    
            arm_offset = 0.0
            arm_angle = 0.0
            if arm_choice == enum_type_to_int(self.arm_types.TUBE_CURVE):
                pass
            elif arm_choice == enum_type_to_int(self.arm_types.TUBE):
                arm_length = np.random.uniform(self.arm_tube_length_min, self.arm_tube_length_max)
                arm_angle = np.deg2rad(np.random.uniform(self.arm_tube_angle_min, self.arm_tube_angle_max))
                arm_N_lines = np.random.randint(self.arm_tube_N_lines_min, self.arm_tube_N_lines_max + 1)
            elif arm_choice == enum_type_to_int(self.arm_types.STICK):
                arm_length_1 = np.random.uniform(self.arm_stick_length_1_min, self.arm_stick_length_1_max)
                arm_length_2 = np.random.uniform(self.arm_stick_length_2_min, self.arm_stick_length_2_max)
                arm_angle = np.deg2rad(np.random.uniform(self.arm_stick_angle_1_min, self.arm_stick_angle_1_max))
                arm_angle_2 = np.deg2rad(np.random.uniform(self.arm_stick_angle_2_min, self.arm_stick_angle_2_max))
                arm_width = np.random.uniform(self.arm_stick_width_min, self.arm_stick_width_max)
                arm_joint_radius = np.max((0.5 * arm_width, np.random.uniform(self.arm_stick_joint_radius_min, self.arm_stick_joint_radius_max)))
                use_arm_joint_point = np.random.random_sample() < self.arm_stick_joint_point_prob
                use_arm_centerline = np.random.random_sample() < self.arm_stick_centerline_prob
                use_arm_tube = np.random.random_sample() < self.arm_stick_tube_prob
                arm_line_width = None
                if use_arm_tube: arm_line_width = np.random.uniform(self.arm_stick_tube_line_width_min, self.arm_stick_tube_line_width_max)
            
            # Shoulder:
            shoulder_type = pick_random_element(self.shoulder_types, self.shoulder_type_probs)
            if np.abs(arm_angle) < np.deg2rad(self.arm_shoulder_angle_max) and shoulder_type != enum_type_to_int(self.shoulder_types.NONE):
                if shoulder_type == enum_type_to_int(self.shoulder_types.RECT):
                    shoulder_width = np.max((np.random.uniform(self.arm_shoulder_width_min, self.arm_shoulder_width_max), self.arm_shoulder_width_to_arm_width_gain_min*arm_width))
                    shoulder_height = np.random.uniform(self.arm_shoulder_height_min, self.arm_shoulder_height_max)
                    shoulder_sketch = generate_shoulder_sketch(shoulder_width, shoulder_height)
                    arm_offset = 0.5 * shoulder_height * np.abs(np.sin(arm_angle))
                elif shoulder_type == enum_type_to_int(self.shoulder_types.CIRCLE):
                    shoulder_radius = np.max((np.random.uniform(self.arm_shoulder_radius_min, self.arm_shoulder_radius_max), 0.5*self.arm_shoulder_width_to_arm_width_gain_min*arm_width))
                    shoulder_width = self.shoulder_width_gain * shoulder_radius
                    shoulder_height = shoulder_radius
                    shoulder_sketch = generate_shoulder_circle_sketch(shoulder_radius)
            else:
                arm_offset = 0.5 * arm_width * np.abs(np.sin(arm_angle))
                shoulder_width, shoulder_height, shoulder_sketch = 0, 0, None
            
            # Hands:
            hand_choice = pick_random_element(self.hand_types, self.hand_type_probs)
            hand_sketch = None
            if hand_choice == enum_type_to_int(self.hand_types.CLAW):
                claw_length_1 = np.random.uniform(self.hand_claw_length_1_min, self.hand_claw_length_1_max)
                claw_length_2 = np.random.uniform(self.hand_claw_length_2_min, self.hand_claw_length_2_max)
                claw_angle_1 = np.random.uniform(self.hand_claw_angle_1_min, self.hand_claw_angle_1_max)
                claw_angle_2 = np.min((np.random.uniform(self.hand_claw_angle_2_min, self.hand_claw_angle_2_max), 0.5 * np.pi - claw_angle_1))
                claw_width = np.random.uniform(self.hand_claw_width_min, self.hand_claw_width_max)
                
                use_claw_circle = np.random.random_sample() < self.hand_claw_joint_prob
                claw_circle_radius = None
                claw_bullet_length = None
                if use_claw_circle:
                    claw_circle_radius = np.max((np.random.uniform(self.hand_claw_circle_radius_min, self.hand_claw_circle_radius_max), self.hand_claw_circle_radius_arm_width_gain_min * arm_width))
                    use_claw_bullet = np.random.random_sample() < self.hand_claw_bullet_prob
                    if use_claw_bullet:
                        claw_bullet_length = np.random.uniform(self.hand_claw_bullet_length_min, self.hand_claw_bullet_length_max)
                        
                use_claw_circle_point = np.random.random_sample() < self.hand_claw_pointy_prob
                    
                use_claw_pointy = np.random.random_sample() < self.hand_claw_pointy_prob
                
                # TODO: make sure angles are not more than 90 deg, circle radius is large enough, angles do not make claw collide
                hand_sketch = generate_claw_sketch(arm_width, claw_width, claw_length_1, claw_length_2, claw_angle_1,
                                                   claw_angle_2, claw_circle_radius, claw_bullet_length, use_claw_circle_point, use_claw_pointy)
            elif hand_choice == enum_type_to_int(self.hand_types.HORSE_SHOE):
                claw_radius = np.random.uniform(self.hand_horse_shoe_radius_min, self.hand_horse_shoe_radius_max)
                claw_width = np.min((np.random.uniform(self.hand_horse_shoe_width_min, self.hand_horse_shoe_width_max),
                                     self.hand_horse_shoe_radius_to_width_gain*claw_radius))
                claw_upper_height = np.random.uniform(self.hand_horse_shoe_upper_height_min, self.hand_horse_shoe_upper_height_max)
                hand_sketch = generate_horse_shoe_sketch(claw_width, claw_upper_height, claw_radius)
            elif hand_choice == enum_type_to_int(self.hand_types.SAWBLADE):
                sawblade_N_blades = np.random.randint(self.hand_sawblade_N_blades_min, self.hand_sawblade_N_blades_max + 1)
                sawblade_flat_angle = np.random.uniform(self.hand_sawblade_flat_angle_min, self.hand_sawblade_flat_angle_max)
                sawblade_rise_angle = np.random.uniform(self.hand_sawblade_rise_angle_min, self.hand_sawblade_rise_angle_max)
                sawblade_lean_angle = np.random.uniform(self.hand_sawblade_lean_angle_min, self.hand_sawblade_lean_angle_max)
                sawblade_radius = np.random.uniform(self.hand_sawblade_radius_min, self.hand_sawblade_radius_max)
                sawblade_height_gain = np.random.uniform(self.hand_sawblade_height_gain_min, self.hand_sawblade_height_gain_max)
                
                sawblade_outer_ring_radius_gain, sawblade_inner_ring_radius_gain = None, None
                use_outer_ring = np.random.random_sample() < self.hand_sawblade_outer_ring_prob
                if use_outer_ring:
                    sawblade_outer_ring_radius_gain = self.hand_sawblade_outer_ring_radius_gain
                    
                use_inner_ring = np.random.random_sample() < self.hand_sawblade_inner_ring_prob
                if use_inner_ring:
                    sawblade_inner_ring_radius_gain = np.random.uniform(self.hand_sawblade_inner_ring_radius_gain_min,
                                                                        self.hand_sawblade_inner_ring_radius_gain_max)
                    
                sawblade_use_rod_center_line = np.random.random_sample() < self.hand_sawblade_rod_center_line_prob
                sawblade_use_hole_points = np.random.random_sample() < self.hand_sawblade_hole_points_prob
                sawblade_hole_radius_gain = np.random.uniform(self.hand_sawblade_hole_radius_gain_min, self.hand_sawblade_hole_radius_gain_max)
                sawblade_rod_width_gain = np.random.uniform(self.hand_sawblade_rod_width_gain_min, self.hand_sawblade_rod_width_gain_max)
                sawblade_flipped = np.random.random_sample() < 0.5
                hand_sketch = generate_sawblade_sketch(sawblade_N_blades, sawblade_radius, sawblade_flat_angle, sawblade_rise_angle,
                                                       sawblade_lean_angle, sawblade_height_gain, sawblade_outer_ring_radius_gain,
                                                       sawblade_inner_ring_radius_gain, sawblade_hole_radius_gain, sawblade_rod_width_gain,
                                                       sawblade_use_hole_points, sawblade_use_rod_center_line, sawblade_flipped)
        
            # Draw arm and hands:                    
            if arm_choice == enum_type_to_int(self.arm_types.TUBE_CURVE):
                x_end = np.random.uniform(self.arm_tube_curve_x_end_min, self.arm_tube_curve_x_end_max)
                y_end = np.random.uniform(self.arm_tube_curve_y_end_min, self.arm_tube_curve_y_end_max)
                x_c1 = np.random.uniform(self.arm_tube_curve_x_c1_min, self.arm_tube_curve_x_c1_max)
                x_c2 = np.random.uniform(self.arm_tube_curve_x_c2_min, self.arm_tube_curve_x_c2_max)
                y_c2 = np.random.uniform(self.arm_tube_curve_y_c2_min, self.arm_tube_curve_y_c2_max)
                if np.random.random_sample() < self.arm_tube_curve_up_prob:
                    y_end *= -1
                arm_sketch = generate_arm_tube_curve_sketch(x_end, y_end, x_c1, 0.0, x_c2, y_c2, width=arm_width,
                                                            N_lines=20, shoulder_sketch=shoulder_sketch, shoulder_width=shoulder_width,
                                                            hand_sketch=hand_sketch, debug=debug)
            elif arm_choice == enum_type_to_int(self.arm_types.TUBE):
                arm_sketch = generate_arm_tube_sketch(arm_length, arm_angle, arm_width, N_lines=arm_N_lines, shoulder_sketch=shoulder_sketch,
                                                      shoulder_width=shoulder_width, hand_sketch=hand_sketch, debug=debug)
            elif arm_choice == enum_type_to_int(self.arm_types.STICK):
                arm_sketch = generate_arm_stick_sketch(arm_length_1, arm_angle, arm_length_2, arm_angle_2, arm_width, arm_joint_radius,
                                                       joint_point=use_arm_joint_point, centerline=use_arm_centerline, tube_line_width=arm_line_width,
                                                       shoulder_sketch=shoulder_sketch, shoulder_width=shoulder_width,
                                                       hand_sketch=hand_sketch, debug=debug)
                
            with vsk.pushMatrix():
                vsk.translate(0.5 * self.body_width - arm_offset, arm_y)  
                vsk.sketch(arm_sketch)
                vsk.translate(-self.body_width + 2 * arm_offset, 0)
                vsk.scale(-1, 1)
                if arm_choice == enum_type_to_int(self.arm_types.TUBE_CURVE) and \
                    np.random.random_sample() < self.arm_tube_curve_flip_prob: vsk.scale(1, -1)
                vsk.sketch(arm_sketch)
                
        
        self.draw_head = np.random.random_sample() < self.head_prob
        
        self.draw_body(vsk, not self.draw_head, debug)
        
        draw_antenna = np.random.random_sample() < self.antenna_prob
        if draw_antenna:
            antenna_base_rect = np.random.random_sample() < self.antenna_rect_base_prob
            antenna_base_width = np.random.uniform(self.antenna_base_width_min, self.antenna_base_width_max)
            antenna_base_height = np.random.uniform(self.antenna_base_height_min, self.antenna_base_height_max)
            antenna_width = np.random.uniform(self.antenna_width_min, self.antenna_width_max)
            antenna_height = np.random.uniform(self.antenna_height_min, self.antenna_height_max)
            antenna_radius = np.random.uniform(self.antenna_radius_min, self.antenna_radius_max)
            antenna_sketch = generate_antenna_sketch(antenna_base_width, antenna_base_height, antenna_width, antenna_height,
                                                     antenna_radius, rect=antenna_base_rect)
        
        if self.draw_head:
            with vsk.pushMatrix():
                vsk.translate(0, -self.body_upper_height)
                use_neck = self.body_choice in (enum_type_to_int(self.body_types.CIRCLE),
                                                enum_type_to_int(self.body_types.BULLET)) or \
                                                    np.random.random_sample() < self.neck_prob
                if use_neck:
                    neck_length = np.random.uniform(self.neck_length_min, self.neck_length_max)
                    neck_width = np.random.uniform(self.neck_width_min, self.neck_width_max)
                    neck_N_lines = np.random.randint(self.neck_N_lines_min, self.neck_N_lines_max + 1)
                    draw_neck(vsk, neck_length, neck_width, neck_N_lines)
                    vsk.translate(0, -neck_length)
                    
                head_choice = pick_random_element(self.head_types, self.head_type_probs)
                if head_choice == enum_type_to_int(self.head_types.RECT):
                    head_width = np.random.uniform(self.head_rect_width_min, self.head_rect_width_max)
                    head_height = np.random.uniform(self.head_rect_height_min, self.head_rect_height_max)
                    head_lower_height = 0.5 * head_height
                    head_upper_height = 0.5 * head_height
                    draw_rect_body(vsk, head_width, head_height, y=-head_lower_height)
                elif head_choice == enum_type_to_int(self.head_types.BULLET):
                    head_width = 2 * np.random.uniform(self.head_bullet_radius_min, self.head_bullet_radius_max)
                    head_lower_height = np.random.uniform(self.head_bullet_lower_height_min, self.head_bullet_lower_height_max)
                    head_upper_height = 0.5 * head_width
                    head_height = head_lower_height + head_upper_height
                    draw_bullet_body(vsk, head_width, head_width, head_lower_height, y=-head_lower_height)
                elif head_choice == enum_type_to_int(self.head_types.TRAPEZOID):
                    head_width = np.random.uniform(self.head_trapezoid_width_min, self.head_trapezoid_width_max)
                    head_upper_width_gain = np.random.uniform(self.head_trapezoid_upper_width_gain_min, self.head_trapezoid_upper_width_gain_max)
                    head_height = np.random.uniform(self.head_trapezoid_height_min, self.head_trapezoid_height_max)
                    head_lower_height = 0.5 * head_height
                    head_upper_height = 0.5 * head_height
                    draw_trapezoid_head(vsk, head_width, head_upper_width_gain, head_height, y=-head_lower_height)
                vsk.translate(0, -head_lower_height)
                
                self.draw_eyes(vsk, head_width, head_height)
                self.draw_mouth(vsk, head_width, head_lower_height, debug=debug)
                    
                # Draw antennas:
                if draw_antenna:
                    vsk.translate(0, -head_upper_height)
                    if head_choice in (enum_type_to_int(self.head_types.BULLET), enum_type_to_int(self.head_types.TRAPEZOID)) \
                        or np.random.random_sample() < self.antenna_single_prob:
                        vsk.sketch(antenna_sketch)
                    else:
                        antenna_x_gain = np.random.uniform(self.antenna_x_gain_min, self.antenna_x_gain_max)
                        antenna_x = np.min((np.max((antenna_x_gain * head_width, 0.5 * antenna_base_width)), 0.5 * (head_width - antenna_base_width)))
                        vsk.translate(antenna_x, 0)
                        vsk.sketch(antenna_sketch)
                        vsk.translate(-2 * antenna_x, 0)
                        vsk.sketch(antenna_sketch)
        else:
            # Draw antennas:
            if draw_antenna:
                with vsk.pushMatrix():
                    vsk.translate(0, -self.body_upper_height)
                    if self.body_choice == enum_type_to_int(self.body_types.BULLET) or np.random.random_sample() < self.antenna_single_prob:
                        vsk.sketch(antenna_sketch)
                    else:
                        antenna_x_gain = np.random.uniform(self.antenna_x_gain_min, self.antenna_x_gain_max)
                        antenna_x = np.min((np.max((antenna_x_gain * self.body_width, 0.5 * antenna_base_width)), 0.5 * (self.body_width - antenna_base_width)))
                        vsk.translate(antenna_x, 0)
                        vsk.sketch(antenna_sketch)
                        vsk.translate(-2 * antenna_x, 0)
                        vsk.sketch(antenna_sketch)
            
            
        vsk.translate(0, self.body_lower_height + leg_length)  # reset position
        
    
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale, self.scale)
        
        self.body_type_probs = np.array([self.body_rect_prob, self.body_circle_prob, self.body_bullet_prob])
        self.head_type_probs = np.array([self.head_rect_prob, self.head_bullet_prob, self.head_trapezoid_prob])
        self.mouth_type_probs = np.array([self.mouth_none_prob, self.mouth_smile_prob, self.mouth_grill_prob, self.mouth_line_prob])
        self.eye_type_probs = np.array([self.eye_point_prob, self.eye_ellipse_point_prob, self.eye_circle_prob, self.eye_rect_prob,
                                        self.eye_circle_single_prob, self.eye_circle_empty_prob])
        self.arm_type_probs = np.array([self.arm_none_prob, self.arm_tube_prob, self.arm_tube_curve_prob, self.arm_stick_prob])
        self.shoulder_type_probs = np.array([self.shoulder_none_prob, self.shoulder_rect_prob, self.shoulder_circle_prob])
        self.hand_type_probs = np.array([self.hand_none_prob, self.hand_claw_prob, self.hand_horse_shoe_prob, self.hand_sawblade_prob])
        self.leg_type_probs = np.array([self.leg_tube_prob, self.leg_omni_prob, self.leg_wheels_prob, self.leg_wheel_prob])
        self.foot_type_probs = np.array([self.foot_shoe_prob, self.foot_arc_prob])
        self.panel_double_type_probs = np.array([self.panel_double_none_prob, self.panel_double_circle_prob, self.panel_double_rect_prob, self.panel_double_line_prob])
        self.panel_double_circle_type_probs = np.array([self.panel_double_circle_single_prob, self.panel_double_circle_filled_prob,
                                                   self.panel_double_circle_striped_prob, self.panel_double_circle_dot_prob])
        self.panel_double_rect_type_probs = np.array([self.panel_double_rect_single_prob, self.panel_double_rect_filled_prob, self.panel_double_rect_striped_prob])
        
        # test_bezier(vsk)
        # test_line_thick(vsk)
        
        # tree = QuadTree(4.0, 5.0, 0.1, 0.075)
        # tree.draw(vsk)
        
        for y in range(self.n_y):
            with vsk.pushMatrix():
                for x in range(self.n_x):
                    self.draw_robot(vsk, debug=self.debug)
                    
                    if self.debug:
                        vsk.stroke(4)
                        vsk.circle(0, 0, radius=0.05)
                        vsk.stroke(1)
                    
                    vsk.translate(self.grid_dist_x, 0)    
            vsk.translate(0, -self.grid_dist_y)
        
        if self.occult:
            vsk.vpype("occult -i")

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    RobotsSketch.display()
    