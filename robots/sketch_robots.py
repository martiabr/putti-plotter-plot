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

def generate_circle_eye_sketch(radius, x_pupil_gain, detail="0.01"):
    eye_sketch = vsketch.Vsketch()
    eye_sketch.detail(detail)
    eye_sketch.circle(0, 0, radius=radius)
    eye_sketch.circle(x_pupil_gain * radius, 0, 1e-2)
    return eye_sketch

def generate_arm_sketch(x_end, y_end, x_c1, y_c1, x_c2, y_c2, width, width_end=None,
                        shoulder_width=None, shoulder_height=None, use_shoulder=False,
                        N_lines=20, detail="0.01", debug=False):
    arm_sketch = vsketch.Vsketch()
    arm_sketch.detail(detail)
    
    if use_shoulder and shoulder_width is not None and shoulder_height is not None:
        arm_sketch.rect(0.5 * shoulder_width, 0, shoulder_width, shoulder_height, mode="center")  
        arm_sketch.translate(shoulder_width, 0)
        
    draw_bezier_thick(arm_sketch, np.array([0.0, 0.0]), np.array([x_end, y_end]), np.array([x_c1, y_c1]),
                        np.array([x_c2, y_c2]), width=width, width_end=width_end, N_segments=40, N_lines=N_lines, debug=debug)
    return arm_sketch


def generate_leg_sketch(x_end, y_end, width, foot_width, foot_height, N_lines=8, detail="0.01", debug=False):
    leg_sketch = vsketch.Vsketch()
    leg_sketch.detail(detail)
    draw_line_thick(leg_sketch, np.array([0.0, 0.0]), np.array([x_end, y_end]), width=width, N_lines=N_lines, debug=debug)
    leg_sketch.translate(x_end, y_end)
    leg_sketch.arc(0, 0, foot_width, foot_height, 0.0, np.pi, close="chord")
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

def draw_rect_body(vsk, width, height, radius=0, draw_inner_rect=False, inner_padding=0.0):
    vsk.rect(0, 0, width, height, radius, mode="center")
    if draw_inner_rect:
        vsk.rect(0, 0, width - inner_padding, height - inner_padding, radius, mode="center")
        
def draw_circle_body(vsk, radius, draw_inner_circle, inner_padding=0.0):
    vsk.circle(0, 0, radius)
    if draw_inner_circle:
        vsk.circle(0, 0, np.max(radius - inner_padding, 0))
    
def draw_bullet_body(vsk, width, arc_height, lower_height):
    bullet_body_shape = vsk.createShape()
    bullet_body_shape.arc(0, 0, width, arc_height, 0, 1.01*np.pi, close="chord", mode="center")  # Note: 1.01 factor is there for shapes to connect
    bullet_body_shape.rect(0, 0.5 * lower_height, width, lower_height, mode="center")
    vsk.shape(bullet_body_shape)
    
    
### Mouths:

def draw_grill_mouth(vsk, width, height, N_lines, debug=False):
    draw_line_thick(vsk, np.array([-0.5 * width, 0]), np.array([0.5 * width, 0]), height, N_lines=N_lines, debug=debug)


class RobotsSketch(vsketch.SketchClass):
    
    # Main parameters:
    n_x = vsketch.Param(3, min_value=1)
    n_y = vsketch.Param(5, min_value=1)
    grid_dist_x = vsketch.Param(8.0)
    grid_dist_y = vsketch.Param(7.0)
    scale = vsketch.Param(0.8, decimals=2)
    
    debug = vsketch.Param(False)
    occult = vsketch.Param(True)
    
    # Body parameters:
    body_types = Enum('BodyType', 'RECT CIRCLE BULLET')
    body_rect_prob = vsketch.Param(0.25, min_value=0)
    body_circle_prob = vsketch.Param(0.25, min_value=0)
    body_bullet_prob = vsketch.Param(0.5, min_value=0)
    
    body_rect_width_max = vsketch.Param(4.0, min_value=0)
    body_rect_width_min = vsketch.Param(1.0, min_value=0)
    body_rect_height_max = vsketch.Param(4.0, min_value=0)
    body_rect_height_min = vsketch.Param(1.0, min_value=0)
    
    inner_rect_prob = vsketch.Param(0.3, min_value=0, max_value=1)
    body_rect_inner_padding_max = vsketch.Param(0.75)
    body_rect_inner_padding_min = vsketch.Param(0.1)
    
    body_circle_radius_max = vsketch.Param(2.5, min_value=0)
    body_circle_radius_min = vsketch.Param(1.2, min_value=0)
    
    inner_circle_prob = vsketch.Param(0.3, min_value=0, max_value=1)
    body_circle_inner_padding_max = vsketch.Param(0.5)
    body_circle_inner_padding_min = vsketch.Param(0.1)
    
    body_bullet_radius_max = vsketch.Param(3.0, min_value=0)
    body_bullet_radius_min = vsketch.Param(0.8, min_value=0)
    body_bullet_lower_height_max = vsketch.Param(2.0, min_value=0)
    body_bullet_lower_height_min = vsketch.Param(0.5, min_value=0)
    
    # Eye parameters:
    eye_types = Enum('EyeType', 'POINT ELLIPSE_POINT CIRCLE')
    eye_point_prob = vsketch.Param(0.25, min_value=0)
    eye_ellipse_point_prob = vsketch.Param(0.25, min_value=0)
    eye_circle_prob = vsketch.Param(0.5, min_value=0)
    
    eye_circle_radius_gain_max = vsketch.Param(0.1, min_value=0)
    eye_circle_x_pupil_gain_max = vsketch.Param(0.2, min_value=0)
    eye_circle_radius_gain_min = vsketch.Param(0.01, min_value=0)
    
    eye_x_gain_max = vsketch.Param(0.6, min_value=0)
    eye_x_gain_min = vsketch.Param(0.1, min_value=0)
    
    # Mouth parameters:
    mouth_types = Enum('MouthType', 'NONE SMILE GRILL')
    mouth_none_prob = vsketch.Param(0.5, min_value=0)
    mouth_smile_prob = vsketch.Param(0.25, min_value=0)
    mouth_grill_prob = vsketch.Param(0.25, min_value=0)
    
    mouth_grill_width_gain_max = vsketch.Param(0.8, min_value=0)
    mouth_grill_width_gain_min = vsketch.Param(0.3, min_value=0)
    mouth_grill_height_max = vsketch.Param(0.6, min_value=0)
    mouth_grill_height_min = vsketch.Param(0.075, min_value=0)
    mouth_grill_N_lines_max = vsketch.Param(8, min_value=0)
    mouth_grill_N_lines_min = vsketch.Param(3, min_value=0)
    
    # Arm parameters:
    arm_shoulder_prob = vsketch.Param(0.5, min_value=0, max_value=1)
    arm_shoulder_width_max = vsketch.Param(0.4, min_value=0)
    arm_shoulder_width_min = vsketch.Param(0.10, min_value=0)
    arm_shoulder_height_max = vsketch.Param(0.8, min_value=0)
    arm_shoulder_height_min = vsketch.Param(0.10, min_value=0)
    
    arm_width_max = vsketch.Param(0.40, min_value=0)
    arm_width_min = vsketch.Param(0.10, min_value=0)
    
    def draw_body(self, vsk, debug=False):
        body_choice = pick_random_element(self.body_types, self.body_type_probs)
        
        if body_choice == enum_type_to_int(self.body_types.RECT):
            self.body_width = np.random.uniform(self.body_rect_width_min, self.body_rect_width_max)
            self.body_height = np.random.uniform(self.body_rect_height_min, self.body_rect_height_max)
            self.body_lower_height = 0.5 * self.body_height
            self.body_upper_height = 0.5 * self.body_height
            
            use_inner_rect = np.random.random_sample() < self.inner_rect_prob
            if use_inner_rect:
                body_inner_padding = np.random.uniform(self.body_rect_inner_padding_min, self.body_rect_inner_padding_max)
            else:
                body_inner_padding = None
                
            draw_rect_body(vsk, self.body_width, self.body_height, draw_inner_rect=use_inner_rect, inner_padding=body_inner_padding)
        elif body_choice == enum_type_to_int(self.body_types.CIRCLE):
            self.body_width = np.random.uniform(self.body_circle_radius_min, self.body_circle_radius_max)
            self.body_height = self.body_width
            self.body_lower_height = 0.5 * self.body_height
            self.body_upper_height = 0.5 * self.body_height
            
            use_inner_circle = np.random.random_sample() < self.inner_circle_prob
            if use_inner_circle:
                body_inner_padding = np.random.uniform(self.body_circle_inner_padding_min, self.body_circle_inner_padding_max)
            else:
                body_inner_padding = None
                
            draw_circle_body(vsk, self.body_width, draw_inner_circle=use_inner_circle, inner_padding=body_inner_padding)
        elif body_choice == enum_type_to_int(self.body_types.BULLET):
            self.body_width = np.random.uniform(self.body_bullet_radius_min, self.body_bullet_radius_max)
            arc_height = np.random.uniform(self.body_bullet_radius_min, self.body_bullet_radius_max)
            self.body_lower_height = np.random.uniform(self.body_bullet_lower_height_min, self.body_bullet_lower_height_max)
            self.body_upper_height = arc_height
            
            draw_bullet_body(vsk, self.body_width, arc_height, self.body_lower_height)
            
        # Eyes:
        eye_choice = pick_random_element(self.eye_types, self.eye_type_probs)
        
        if eye_choice == enum_type_to_int(self.eye_types.POINT):
            eye_sketch = generate_point_eye_sketch()
        elif eye_choice == enum_type_to_int(self.eye_types.ELLIPSE_POINT):
            eye_sketch = generate_ellipse_point_eye_sketch()
        elif eye_choice == enum_type_to_int(self.eye_types.CIRCLE):
            eye_radius = np.random.uniform(self.eye_circle_radius_gain_min, self.eye_circle_radius_gain_max) * self.body_width
            pupil_x_gain = np.random.uniform(-self.eye_circle_x_pupil_gain_max, self.eye_circle_x_pupil_gain_max)
            eye_sketch = generate_circle_eye_sketch(eye_radius, pupil_x_gain)
            
        eye_x_gain = np.random.uniform(self.eye_x_gain_min, self.eye_x_gain_max)
        with vsk.pushMatrix():
            vsk.translate(eye_x_gain * 0.5 * self.body_width, 0)
            vsk.sketch(eye_sketch)
            vsk.translate(-eye_x_gain * self.body_width, 0)
            if np.random.random_sample() < 0.5: vsk.scale(-1, 1)
            vsk.sketch(eye_sketch)
        
        
        
        # Mouth:
        mouth_choice = pick_random_element(self.mouth_types, self.mouth_type_probs)
        with vsk.pushMatrix():
            vsk.translate(0, 0.2 * self.body_lower_height)
            if mouth_choice == enum_type_to_int(self.mouth_types.SMILE):
                draw_bezier(vsk, np.array([-0.1, 0]), np.array([0.1, 0]), np.array([-0.1, 0.2]), np.array([0.1, 0.2]), debug=debug)  # TODO: function of this
            elif mouth_choice == enum_type_to_int(self.mouth_types.GRILL):
                mouth_width = np.random.uniform(self.mouth_grill_width_gain_min, self.mouth_grill_width_gain_max) * self.body_width
                mouth_height = np.random.uniform(self.mouth_grill_height_min, self.mouth_grill_height_max)
                mouth_N_lines = np.random.randint(self.mouth_grill_N_lines_min, self.mouth_grill_N_lines_max + 1)
                vsk.translate(0, 0.5 * mouth_height)
                draw_grill_mouth(vsk, mouth_width, mouth_height, mouth_N_lines, debug=debug)

    def draw_robot(self, vsk, debug=False):
        self.draw_body(vsk, debug)
        
        # Arms:
        use_shoulder = np.random.random_sample() < self.arm_shoulder_prob
        if use_shoulder:
            shoulder_width = np.random.uniform(self.arm_shoulder_width_min, self.arm_shoulder_width_max)
            shoulder_height = np.random.uniform(self.arm_shoulder_height_min, self.arm_shoulder_height_max)
        else:
            shoulder_width, shoulder_height = None, None
            
        arm_width = np.random.uniform(self.arm_width_min, self.arm_width_max)
        
        arm_sketch = generate_arm_sketch(1.0, -0.75, 0.8, 0.0, 1.0, -0.1, width=arm_width,
                                         shoulder_width=shoulder_width, shoulder_height=shoulder_height,
                                         use_shoulder=use_shoulder, N_lines=20, debug=debug)
        with vsk.pushMatrix():
            vsk.translate(0.5 * self.body_width, 0)  
            vsk.sketch(arm_sketch)
            vsk.translate(-self.body_width, 0)
            vsk.scale(-1, 1)
            vsk.sketch(arm_sketch)
            
        
        # Legs: 
        leg_sketch = generate_leg_sketch(0, 0.6, 0.3, 0.5, 0.4, debug=debug)
        with vsk.pushMatrix():
            vsk.translate(0.35 * self.body_width, self.body_lower_height)
            vsk.sketch(leg_sketch)
            vsk.translate(-2 * 0.35 * self.body_width, 0)
            vsk.sketch(leg_sketch)
        
        # Draw antennas:
        # antenna_sketch = generate_antenna_sketch(0.25, 0.15, 0.075, 0.6, 0.2)
        # with vsk.pushMatrix():
        #     vsk.translate(0.3 * self.body_width, -self.body_upper_height - 0.5 * 0.15)
        #     vsk.sketch(antenna_sketch)
        #     vsk.translate(-2 * 0.3 * self.body_width, 0)
        #     vsk.sketch(antenna_sketch)
        
    
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale, self.scale)
        
        self.body_type_probs = np.array([self.body_rect_prob, self.body_circle_prob, self.body_bullet_prob])
        self.mouth_type_probs = np.array([self.mouth_none_prob, self.mouth_smile_prob, self.mouth_grill_prob])
        self.eye_type_probs = np.array([self.eye_point_prob, self.eye_ellipse_point_prob, self.eye_circle_prob])
        
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
