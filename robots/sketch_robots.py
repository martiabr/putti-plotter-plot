import vsketch
import numpy as np
from enum import Enum
from curve_utils import draw_bezier, draw_line_thick, draw_bezier_thick

# - Should work such that I can call a series of functions and then draw any robot I want. 
#   Allows to create a drawing of every type possible, e.g. show every possible mouth, eyes, arms...
#   To have an overview.
# - Maybe have two scripts - robot editor, and random robot grid generator 


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
    eye_sketch.circle(0, 0, 1e-2)
    return eye_sketch

def generate_ellipse_point_eye_sketch(detail="0.01"):
    eye_sketch = vsketch.Vsketch()
    eye_sketch.fill(1)
    eye_sketch.detail(detail)
    eye_sketch.ellipse(0, 0, 0.05, 0.075)
    eye_sketch.noFill()
    return eye_sketch

def generate_circle_eye_sketch(radius, x_pupil_gain=None, detail="0.01"):
    eye_sketch = vsketch.Vsketch()
    eye_sketch.detail(detail)
    eye_sketch.circle(0, 0, radius=radius)
    if x_pupil_gain is not None: eye_sketch.circle(x_pupil_gain * radius, 0, 1e-2)
    return eye_sketch

def generate_claw_sketch(base_width, claw_width, length_1, length_2, angle_1, angle_2, joint_radius=None,
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
        hand_sketch.circle(0.5 * base_width, 0, radius=joint_radius)
        if joint_point:
            hand_sketch.circle(.5 * base_width, 0, 1e-2)
            
    
    return hand_sketch


def generate_shoulder_sketch(width, height, detail="0.01"):
    shoulder_sketch = vsketch.Vsketch()
    shoulder_sketch.detail(detail)
    shoulder_sketch.rect(0.5 * width, 0, width, height, mode="center")  
    shoulder_sketch.translate(width, 0)
    return shoulder_sketch


def generate_arm_stick_sketch(link_length_1, alpha_1, link_length_2, alpha_2, width, joint_radius,
                              joint_point=False, shoulder_sketch=None, shoulder_width=None, hand_sketch=None,
                              detail="0.01", debug=False, debug_radius=0.05):
    arm_sketch = vsketch.Vsketch()
    arm_sketch.detail(detail)
    
    with arm_sketch.pushMatrix():
        arm_sketch.rotate(alpha_1)
        if shoulder_sketch is not None and shoulder_width is not None:
            arm_sketch.sketch(shoulder_sketch)
            arm_sketch.translate(shoulder_width, 0)
            
        arm_sketch.rect(0.5 * link_length_1, 0, link_length_1, width, mode="center")
        if debug:
            arm_sketch.stroke(2)
            arm_sketch.circle(0, 0, radius=debug_radius)
            arm_sketch.circle(link_length_1, 0, radius=debug_radius)
            arm_sketch.line(0, 0, link_length_1, 0)
            arm_sketch.stroke(1)
            
        arm_sketch.translate(link_length_1, 0)
        arm_sketch.rotate(alpha_2 - alpha_1)
        arm_sketch.rect(0.5 * link_length_2, 0, link_length_2, width, mode="center")
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
        
    return arm_sketch


def generate_arm_tube_sketch(length, alpha, width, N_lines=20, shoulder_sketch=None, shoulder_width=None,
                             hand_sketch=None, detail="0.01", debug=False):
    arm_sketch = vsketch.Vsketch()
    arm_sketch.detail(detail)
    
    with arm_sketch.pushMatrix():
        arm_sketch.rotate(alpha)
        if shoulder_sketch is not None and shoulder_width is not None:
            arm_sketch.sketch(shoulder_sketch)
            arm_sketch.translate(shoulder_width, 0)
        
        draw_line_thick(arm_sketch, np.array([0, 0]), np.array([length, 0]), width, N_lines=N_lines, debug=debug)
    
        if hand_sketch is not None:
            arm_sketch.translate(length, 0)
            arm_sketch.sketch(hand_sketch)
            
    return arm_sketch
    

def generate_arm_tube_curve_sketch(x_end, y_end, x_c1, y_c1, x_c2, y_c2, width, width_end=None,
                        N_lines=20, shoulder_sketch=None, shoulder_width=None, hand_sketch=None, detail="0.01", debug=False):
    arm_sketch = vsketch.Vsketch()
    arm_sketch.detail(detail)
    
    if shoulder_sketch is not None and shoulder_width is not None:
        arm_sketch.sketch(shoulder_sketch)
        arm_sketch.translate(shoulder_width, 0)
        
    draw_bezier_thick(arm_sketch, np.array([0.0, 0.0]), np.array([x_end, y_end]), np.array([x_c1, y_c1]),
                        np.array([x_c2, y_c2]), width=width, width_end=width_end, N_segments=40, N_lines=N_lines, debug=debug)
    
    if hand_sketch is not None:
        angle = np.arctan2(y_end - y_c2, x_end - x_c2)
        arm_sketch.translate(x_end, y_end)
        arm_sketch.rotate(angle)
        arm_sketch.sketch(hand_sketch)
            
    return arm_sketch


def generate_leg_tube_sketch(length, width, N_lines=8, detail="0.01", debug=False):
    leg_sketch = vsketch.Vsketch()
    leg_sketch.detail(detail)
    draw_line_thick(leg_sketch, np.array([0.0, 0.0]), np.array([0, -length]), width=width, N_lines=N_lines, debug=debug)
    return leg_sketch

def generate_leg_omni_sketch(width, trapezoid_width, height, radius, radius_inner, detail="0.01"):
    width_lower = trapezoid_width
    leg_sketch = vsketch.Vsketch()
    leg_sketch.detail(detail)
    leg_sketch.circle(0, -radius, radius=radius)
    leg_sketch.circle(0, -radius, radius=radius_inner)
    leg_sketch.translate(0, -radius)
    leg_sketch.polygon([(0.5 * width_lower, 0), (0.5 * width, -height), (-0.5 * width, -height), (-0.5 * width_lower, 0)], close=True)
    return leg_sketch


def generate_antenna_sketch(base_width, base_height, antenna_width, antenna_height, antenna_radius, detail="0.01"):
    antenna_sketch = vsketch.Vsketch()
    antenna_sketch.detail(detail)
    antenna_sketch.rect(0, 0, base_width, base_height, mode="center")
    antenna_sketch.translate(0, -0.5 * (base_height + antenna_height))
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

def draw_grill_mouth(vsk, width, height, N_lines, debug=False):
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
    
    inner_circle_prob = vsketch.Param(0.3, min_value=0, max_value=1)
    body_circle_inner_padding_min = vsketch.Param(0.2)
    body_circle_inner_padding_max = vsketch.Param(0.5)
    
    body_bullet_radius_min = vsketch.Param(0.75, min_value=0)
    body_bullet_radius_max = vsketch.Param(1.5, min_value=0)
    body_bullet_lower_height_min = vsketch.Param(0.5, min_value=0)
    body_bullet_lower_height_max = vsketch.Param(2.0, min_value=0)
    
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
    eye_types = Enum('EyeType', 'POINT ELLIPSE_POINT CIRCLE SINGLE_CIRCLE CIRCLE_EMPTY')
    eye_point_prob = vsketch.Param(0.2, min_value=0, max_value=1)
    eye_ellipse_point_prob = vsketch.Param(0.2, min_value=0, max_value=1)
    eye_circle_prob = vsketch.Param(0.2, min_value=0, max_value=1)
    eye_circle_single_prob = vsketch.Param(0.2, min_value=0, max_value=1)
    eye_circle_empty_prob = vsketch.Param(0.2, min_value=0, max_value=1)
    
    eye_circle_radius_gain_min = vsketch.Param(0.06, min_value=0)
    eye_circle_radius_gain_max = vsketch.Param(0.1, min_value=0)
    eye_single_circle_radius_gain_min = vsketch.Param(0.08, min_value=0)
    eye_single_circle_radius_gain_max = vsketch.Param(0.15, min_value=0)
    eye_circle_empty_radius_gain_min = vsketch.Param(0.01, min_value=0)
    eye_circle_empty_radius_gain_max = vsketch.Param(0.03, min_value=0)
    
    eye_circle_x_pupil_gain_max = vsketch.Param(0.2, min_value=0)
    
    eye_x_gain_min = vsketch.Param(0.1, min_value=0)
    eye_x_gain_max = vsketch.Param(0.6, min_value=0)
    
    # Mouth parameters:
    mouth_types = Enum('MouthType', 'NONE SMILE GRILL')
    mouth_none_prob = vsketch.Param(0.3, min_value=0)
    mouth_smile_prob = vsketch.Param(0.3, min_value=0)
    mouth_grill_prob = vsketch.Param(0.4, min_value=0)
    
    mouth_grill_width_gain_min = vsketch.Param(0.3, min_value=0)
    mouth_grill_width_gain_max = vsketch.Param(0.8, min_value=0)
    mouth_grill_height_min = vsketch.Param(0.075, min_value=0)
    mouth_grill_height_max = vsketch.Param(0.5, min_value=0)
    mouth_grill_N_lines_min = vsketch.Param(3, min_value=0)
    mouth_grill_N_lines_max = vsketch.Param(8, min_value=0)
    
    # Arm parameters:
    arm_types = Enum('ArmType', 'NONE TUBE TUBE_CURVE STICK')
    arm_none_prob = vsketch.Param(0.0, min_value=0)
    arm_tube_prob = vsketch.Param(0.2, min_value=0)
    arm_tube_curve_prob = vsketch.Param(0.3, min_value=0)
    arm_stick_prob = vsketch.Param(0.5, min_value=0)
    
    arm_y_gain_max = vsketch.Param(0.4, min_value=0)
    
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
    
    # Shoulder parameters:
    arm_shoulder_prob = vsketch.Param(0.5, min_value=0, max_value=1)
    arm_shoulder_width_min = vsketch.Param(0.20, min_value=0)
    arm_shoulder_width_max = vsketch.Param(0.5, min_value=0)
    arm_shoulder_height_min = vsketch.Param(0.20, min_value=0)
    arm_shoulder_height_max = vsketch.Param(0.6, min_value=0)
    arm_shoulder_angle_max = vsketch.Param(20, min_value=0)  # +- outside this deg angle we will not draw shoulders bause it will look weird
    
    # Hand parameters:
    hand_types = Enum('FootType', 'NONE CLAW')
    hand_none_prob = vsketch.Param(0.5, min_value=0)
    hand_claw_prob = vsketch.Param(0.5, min_value=0)
    
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
    hand_claw_circle_prob = vsketch.Param(0.5, min_value=0)
    hand_claw_circle_point_prob = vsketch.Param(0.5, min_value=0)
    hand_claw_circle_radius_min = vsketch.Param(0.075, min_value=0)
    hand_claw_circle_radius_max = vsketch.Param(0.15, min_value=0)
    hand_claw_pointy_prob = vsketch.Param(0.3, min_value=0)

    # Leg parameters:
    leg_types = Enum('LegType', 'TUBE OMNI')
    leg_tube_prob = vsketch.Param(0.5, min_value=0)
    leg_omni_prob = vsketch.Param(0.5, min_value=0)
    
    leg_x_gain_max = vsketch.Param(0.40, min_value=0)
    leg_x_gain_min = vsketch.Param(0.15, min_value=0)
    
    leg_tube_width_max = vsketch.Param(0.40, min_value=0)
    leg_tube_width_min = vsketch.Param(0.1, min_value=0)
    leg_tube_length_min = vsketch.Param(0.4, min_value=0)
    leg_tube_length_max = vsketch.Param(1.75, min_value=0)
    leg_tube_N_lines_min =  vsketch.Param(0, min_value=0)
    leg_tube_N_lines_max =  vsketch.Param(20, min_value=0)
    
    leg_omni_width_gain_min = vsketch.Param(0.5, min_value=0, max_value=1)
    leg_omni_width_gain_max = vsketch.Param(0.9, min_value=0, max_value=1)
    leg_omni_trapezoid_width_gain_min = vsketch.Param(0.5, min_value=0, max_value=1)
    leg_omni_trapezoid_width_gain_max = vsketch.Param(0.7, min_value=0, max_value=1)
    leg_omni_height_gain_min = vsketch.Param(0.15, min_value=0, max_value=1)
    leg_omni_height_gain_max = vsketch.Param(0.4, min_value=0, max_value=1)
    leg_omni_radius_gain_min = vsketch.Param(0.1, min_value=0, max_value=1)
    leg_omni_radius_gain_max = vsketch.Param(0.5, min_value=0, max_value=1)
    leg_omni_inner_radius_min = vsketch.Param(0.05, min_value=0, max_value=1)
    leg_omni_inner_radius_max = vsketch.Param(0.1, min_value=0, max_value=1)
    
    # Foot parameters:
    foot_types = Enum('FootType', 'RECT ARC')
    foot_rect_prob = vsketch.Param(0.5, min_value=0)
    foot_arc_prob = vsketch.Param(0.5, min_value=0)
    
    def draw_eyes(self, vsk, face_width):
        self.eye_choice = pick_random_element(self.eye_types, self.eye_type_probs)
        self.eye_radius = 0.0
        
        if self.eye_choice == enum_type_to_int(self.eye_types.POINT):
            eye_sketch = generate_point_eye_sketch()
        elif self.eye_choice == enum_type_to_int(self.eye_types.ELLIPSE_POINT):
            eye_sketch = generate_ellipse_point_eye_sketch()
        elif self.eye_choice == enum_type_to_int(self.eye_types.CIRCLE):
            self.eye_radius = np.random.uniform(self.eye_circle_radius_gain_min, self.eye_circle_radius_gain_max) * self.body_width
            pupil_x_gain = np.random.uniform(-self.eye_circle_x_pupil_gain_max, self.eye_circle_x_pupil_gain_max)
            eye_sketch = generate_circle_eye_sketch(self.eye_radius, pupil_x_gain)
        elif self.eye_choice == enum_type_to_int(self.eye_types.SINGLE_CIRCLE):
            self.eye_radius = np.random.uniform(self.eye_single_circle_radius_gain_min, self.eye_single_circle_radius_gain_max) * self.body_width
            pupil_x_gain = np.random.uniform(-self.eye_circle_x_pupil_gain_max, self.eye_circle_x_pupil_gain_max)
            eye_sketch = generate_circle_eye_sketch(self.eye_radius, pupil_x_gain)
        elif self.eye_choice == enum_type_to_int(self.eye_types.CIRCLE_EMPTY):
            self.eye_radius = np.random.uniform(self.eye_circle_empty_radius_gain_min, self.eye_circle_empty_radius_gain_max) * self.body_width
            eye_sketch = generate_circle_eye_sketch(self.eye_radius)
            
        
        if self.eye_choice == enum_type_to_int(self.eye_types.SINGLE_CIRCLE):
            vsk.sketch(eye_sketch)
        else:
            eye_x_gain = np.random.uniform(self.eye_x_gain_min, self.eye_x_gain_max)
            with vsk.pushMatrix():
                x_eye = np.max((eye_x_gain * 0.5 * face_width, 1.2 * self.eye_radius))
                vsk.translate(x_eye, 0)
                vsk.sketch(eye_sketch)
                vsk.translate(-2 * x_eye, 0)
                if np.random.random_sample() < 0.5: vsk.scale(-1, 1)
                vsk.sketch(eye_sketch)
                
                
    def draw_mouth(self, vsk, face_width, face_lower_height, debug=False):
        mouth_choice = pick_random_element(self.mouth_types, self.mouth_type_probs)
        with vsk.pushMatrix():
            vsk.translate(0, 0.2 * face_lower_height + self.eye_radius)
            
            if mouth_choice == enum_type_to_int(self.mouth_types.SMILE):
                draw_bezier(vsk, np.array([-0.1, 0]), np.array([0.1, 0]), np.array([-0.1, 0.2]), np.array([0.1, 0.2]), debug=debug)  # TODO: function of this
            elif mouth_choice == enum_type_to_int(self.mouth_types.GRILL):
                mouth_width = np.random.uniform(self.mouth_grill_width_gain_min, self.mouth_grill_width_gain_max) * face_width
                mouth_height = np.random.uniform(self.mouth_grill_height_min, self.mouth_grill_height_max)
                mouth_N_lines = np.random.randint(self.mouth_grill_N_lines_min, self.mouth_grill_N_lines_max + 1)
                vsk.translate(0, 0.5 * mouth_height)
                draw_grill_mouth(vsk, mouth_width, mouth_height, mouth_N_lines, debug=debug)

    
    def draw_body(self, vsk, draw_face=False, debug=False):
        if self.body_choice == enum_type_to_int(self.body_types.RECT):
            use_inner_rect = np.random.random_sample() < self.inner_rect_prob
            if use_inner_rect:
                body_inner_padding = np.random.uniform(self.body_rect_inner_padding_min, self.body_rect_inner_padding_max)
            else:
                body_inner_padding = None
                
            draw_rect_body(vsk, self.body_width, self.body_height, draw_inner_rect=use_inner_rect, inner_padding=body_inner_padding)
        elif self.body_choice == enum_type_to_int(self.body_types.CIRCLE):
            use_inner_circle = np.random.random_sample() < self.inner_circle_prob
            if use_inner_circle:
                body_inner_padding = np.random.uniform(self.body_circle_inner_padding_min, self.body_circle_inner_padding_max)
            else:
                body_inner_padding = None
                
            draw_circle_body(vsk, self.body_width, draw_inner_circle=use_inner_circle, inner_padding=body_inner_padding)
        elif self.body_choice == enum_type_to_int(self.body_types.BULLET):
            draw_bullet_body(vsk, self.body_width, self.body_width, self.body_lower_height)
            
        if draw_face:
            self.draw_eyes(vsk, self.body_width)
            self.draw_mouth(vsk, self.body_width, self.body_lower_height)
        else:
            # TODO: Add control panels and fun flurishes here!
            pass

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
        
        
        # Legs: 
        leg_choice = pick_random_element(self.leg_types, self.leg_type_probs)
        
        if leg_choice == enum_type_to_int(self.leg_types.TUBE):  # two
            leg_x_gain = np.random.uniform(self.leg_x_gain_min, self.leg_x_gain_max)
        else:  # one
            pass
        
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
            
            leg_length = leg_omni_radius + leg_omni_height
            leg_sketch = generate_leg_omni_sketch(leg_omni_width, leg_omni_trapezoid_width, leg_omni_height, leg_omni_radius, leg_omni_inner_radius)
        
        # Foot:
        if leg_choice == enum_type_to_int(self.leg_types.TUBE):
            leg_sketch.arc(0, 0, 0.5, 0.4, 0, np.pi, close="chord")
            
        with vsk.pushMatrix():
            if leg_choice == enum_type_to_int(self.leg_types.TUBE):  # two
                vsk.translate(leg_x_gain * self.body_width, 0)
                vsk.sketch(leg_sketch)
                vsk.translate(-2 * leg_x_gain * self.body_width, 0)
                vsk.sketch(leg_sketch)
            else:
                vsk.sketch(leg_sketch)
        vsk.translate(0, -self.body_lower_height - leg_length)
        
               
        # Arms:
        arm_choice = pick_random_element(self.arm_types, self.arm_type_probs)
        if arm_choice != enum_type_to_int(self.arm_types.NONE):            
            arm_y = 0.0
            if self.body_choice == enum_type_to_int(self.body_types.BULLET):
                arm_y = self.body_lower_height * np.random.uniform(0, self.arm_y_gain_max)
            elif self.body_choice == enum_type_to_int(self.body_types.RECT):
                arm_y = self.body_height * (np.random.uniform(0, self.arm_y_gain_max) - 0.5 * self.arm_y_gain_max)
                
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
                use_arm_joint_point =  np.random.random_sample() < self.arm_stick_joint_point_prob
            
            use_shoulder = np.random.random_sample() < self.arm_shoulder_prob and np.abs(arm_angle) < np.deg2rad(self.arm_shoulder_angle_max)
            if use_shoulder:
                shoulder_width = np.random.uniform(self.arm_shoulder_width_min, self.arm_shoulder_width_max)
                shoulder_height = np.random.uniform(self.arm_shoulder_height_min, self.arm_shoulder_height_max)
                shoulder_sketch = generate_shoulder_sketch(shoulder_width, shoulder_height)
                arm_offset = 0.5 * shoulder_height * np.abs(np.sin(arm_angle))
            else:
                arm_offset = 0.5 * arm_width * np.abs(np.sin(arm_angle))
                shoulder_width, shoulder_height, shoulder_sketch = None, None, None
            
            # Hands:
            hand_choice = pick_random_element(self.hand_types, self.hand_type_probs)
            hand_sketch = None
            if hand_choice == enum_type_to_int(self.hand_types.CLAW):
                claw_length_1 = np.random.uniform(self.hand_claw_length_1_min, self.hand_claw_length_1_max)
                claw_length_2 = np.random.uniform(self.hand_claw_length_2_min, self.hand_claw_length_2_max)
                claw_angle_1 = np.random.uniform(self.hand_claw_angle_1_min, self.hand_claw_angle_1_max)
                claw_angle_2 = np.min((np.random.uniform(self.hand_claw_angle_2_min, self.hand_claw_angle_2_max), 0.5 * np.pi - claw_angle_1))
                claw_width = np.random.uniform(self.hand_claw_width_min, self.hand_claw_width_max)
                use_claw_circle = np.random.random_sample() < self.hand_claw_circle_prob
                claw_circle_radius = None
                if use_claw_circle:
                    claw_circle_radius = np.random.uniform(self.hand_claw_circle_radius_min, self.hand_claw_circle_radius_max)
                use_claw_circle_point = np.random.random_sample() < self.hand_claw_pointy_prob
                    
                use_claw_pointy = np.random.random_sample() < self.hand_claw_pointy_prob
                
                # TODO: make sure angles are not more than 90 deg, circle radius is large enough, angles do not make claw collide
                hand_sketch = generate_claw_sketch(arm_width, claw_width, claw_length_1, claw_length_2, claw_angle_1,
                                                   claw_angle_2, claw_circle_radius, use_claw_circle_point, use_claw_pointy)
        
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
                                                       joint_point=use_arm_joint_point, shoulder_sketch=shoulder_sketch, shoulder_width=shoulder_width,
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
                    draw_bullet_body(vsk, head_width, head_width, head_lower_height, y=-head_lower_height)
                elif head_choice == enum_type_to_int(self.head_types.TRAPEZOID):
                    head_width = np.random.uniform(self.head_trapezoid_width_min, self.head_trapezoid_width_max)
                    head_upper_width_gain = np.random.uniform(self.head_trapezoid_upper_width_gain_min, self.head_trapezoid_upper_width_gain_max)
                    head_height = np.random.uniform(self.head_trapezoid_height_min, self.head_trapezoid_height_max)
                    head_lower_height = 0.5 * head_height
                    head_upper_height = 0.5 * head_height
                    draw_trapezoid_head(vsk, head_width, head_upper_width_gain, head_height, y=-head_lower_height)
                vsk.translate(0, -head_lower_height)
                
                self.draw_eyes(vsk, head_width)
                self.draw_mouth(vsk, head_width, head_lower_height)
                    
                # Draw antennas:
                # antenna_sketch = generate_antenna_sketch(0.25, 0.15, 0.075, 0.6, 0.2)
                # with vsk.pushMatrix():
                #     vsk.translate(0.3 * head_width, -head_upper_height - 0.5 * 0.15)
                #     vsk.sketch(antenna_sketch)
                #     vsk.translate(-2 * 0.3 * head_width, 0)
                #     vsk.sketch(antenna_sketch)
        
        vsk.translate(0, self.body_lower_height + leg_length)  # reset position
        
    
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale, self.scale)
        
        self.body_type_probs = np.array([self.body_rect_prob, self.body_circle_prob, self.body_bullet_prob])
        self.head_type_probs = np.array([self.head_rect_prob, self.head_bullet_prob, self.head_trapezoid_prob])
        self.mouth_type_probs = np.array([self.mouth_none_prob, self.mouth_smile_prob, self.mouth_grill_prob])
        self.eye_type_probs = np.array([self.eye_point_prob, self.eye_ellipse_point_prob, self.eye_circle_prob,
                                        self.eye_circle_single_prob, self.eye_circle_empty_prob])
        self.arm_type_probs = np.array([self.arm_none_prob, self.arm_tube_prob, self.arm_tube_curve_prob, self.arm_stick_prob])
        self.hand_type_probs = np.array([self.hand_none_prob, self.hand_claw_prob])
        self.leg_type_probs = np.array([self.leg_tube_prob, self.leg_omni_prob])
        self.foot_type_probs = np.array([self.foot_rect_prob, self.foot_arc_prob])
        
        # test_bezier(vsk)
        # test_line_thick(vsk)
        
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
