from numpy.core.numeric import outer
import vsketch
import numpy as np
from enum import Enum

# - random small details like circles and rectangles (chance of drawing if not fill)
# - random square on outside 
# - random part that goes in on both sides
# - fins (either just triangle or parallelogram) (either at the bottom or somewhere higher up)
# - side boosters (could also have 4 boosters, so also one in front)
# - star ship easter egg?
# - dome shaped nose cone?
# - improve pointy tip
# - improve body building algorithm - not too large jumps!!!!!

class RocketSketch(vsketch.SketchClass):
    
    ### Main params:
    
    n_x = vsketch.Param(18)
    n_y = vsketch.Param(6)
    grid_dist_x = vsketch.Param(4.0)
    grid_dist_y = vsketch.Param(18.0)
    scale = vsketch.Param(0.22, decimals=2)
    
    ### Engine params:
    
    min_engines = vsketch.Param(1)
    max_engines = vsketch.Param(5)
    
    prob_trial_engine = vsketch.Param(0.4)  # p in geometric distribution
    
    min_inner_engine_width_gain = vsketch.Param(0.2)
    max_inner_engine_width_gain = vsketch.Param(0.8)
    
    min_engine_height_gain = vsketch.Param(0.5)
    max_engine_height_gain = vsketch.Param(1.2)
    
    min_bottom_width = vsketch.Param(1.0)
    max_bottom_width = vsketch.Param(2.0)
    
    ### Body params:
    
    min_body_segments = vsketch.Param(4)
    max_body_segments = vsketch.Param(7)
    
    min_body_width = vsketch.Param(0.4)
    max_body_segment_reduction = vsketch.Param(0.5)
    
    min_body_height = vsketch.Param(0.4)
    max_body_height = vsketch.Param(2.0)
    
    min_trapezoid_height = vsketch.Param(0.3)
    max_trapezoid_height = vsketch.Param(0.8)
    
    prob_trapezoid = vsketch.Param(0.9)
    
    prob_same_width = vsketch.Param(0.5)
    prob_smaller_width = vsketch.Param(0.5)
    
    prob_filled_parts = vsketch.Param(0.7)
    prob_fill = vsketch.Param(0.3)
    min_fill_line_padding = vsketch.Param(0.15)
    max_fill_line_padding = vsketch.Param(0.3)
    
    ### Tip params:
    
    tip_types = Enum('TipType', 'ellipse_pointy ellipse triangle trapezoid_fancy')
    prob_tips = [0.25, 0.25, 0.25, 0.25]
    
    min_tip_height = vsketch.Param(0.8)
    max_tip_height = vsketch.Param(2.5)
    
    min_trapezoid_tip_gain = vsketch.Param(0.1)
    max_trapezoid_tip_gain = vsketch.Param(0.2)
    
    def draw_rocket_tip_ellipse_pointy(self, vsk, width):
        angle = np.random.uniform(0, np.pi / 2)  # TODO: parametrize
        height = np.random.uniform(self.min_tip_height, self.max_tip_height)
        a = 0.5 * width / (1 - np.cos(angle))
        b = 0.5 * height / np.sin(angle)
        x = a*np.cos(angle)
        vsk.arc(- x, 0, 2 * a, 2 * b, 0, angle)
        vsk.arc(x, 0, 2 * a, 2 * b, np.pi - angle, np.pi)
        
    def draw_rocket_tip_ellipse(self, vsk, width):
        height = np.random.uniform(self.min_tip_height, self.max_tip_height)
        vsk.arc(0, 0, width, height, 0, np.pi)
    
    def draw_rocket_tip_triangle(self, vsk, width):
        height = np.random.uniform(self.min_tip_height, self.max_tip_height)
        vsk.triangle(-0.5*width, 0, 0.5*width, 0, 0, -height)
        
    def draw_rocket_tip_trapezoid_fancy(self, vsk, width):
        height = np.random.uniform(self.min_tip_height, self.max_tip_height)
        upper_width = np.random.uniform(self.min_trapezoid_tip_gain, self.max_trapezoid_tip_gain) * width
        vsk.polygon([(-0.5*width, 0), (0.5*width, 0), (0.5*upper_width, -height), (-0.5*upper_width, -height)], close=True)
        
        vsk.translate(0, -height)
        
        max_trapezoid_fancy_tip = 4
        n_tip = np.random.randint(1, max_trapezoid_fancy_tip)
        for i in range(n_tip):
            height_i = np.random.uniform(0.2, 0.6)
            vsk.polygon([(-0.5*upper_width, 0), (0.5*upper_width, 0), (0.5*upper_width, -height_i), (-0.5*upper_width, -height_i)], close=True)
            vsk.translate(0, -height_i)
            # TODO: improve this process with varying widths
    
    def draw_rocket_tip(self, vsk, width):
        choice = np.random.choice(len(self.tip_types), p=self.prob_tips)
        if choice == self.tip_types.ellipse_pointy.value - 1:
            self.draw_rocket_tip_ellipse_pointy(vsk, width)
        elif choice == self.tip_types.ellipse.value - 1:
            self.draw_rocket_tip_ellipse(vsk, width)
        elif choice == self.tip_types.triangle.value - 1:
            self.draw_rocket_tip_triangle(vsk, width)
        elif choice == self.tip_types.trapezoid_fancy.value - 1:
            self.draw_rocket_tip_trapezoid_fancy(vsk, width)
    
    def draw_rocket_extras(self, vsk):
        pass
    
    def draw_rocket_body(self, vsk, bottom_width):
        # some prob the same width, some prob smaller
        n_body_segments = np.random.randint(self.min_body_segments, self.max_body_segments + 1)
        widths = np.zeros(n_body_segments)
        heights = np.zeros(n_body_segments)
        choices = np.zeros(n_body_segments, dtype=int)
        
        use_filled_parts = np.random.uniform() < self.prob_filled_parts
        
        for i in range(n_body_segments):
            if i == 0:
                choices[0] = 1
                widths[i] = bottom_width
            else:
                choice = np.random.choice(2, p=[self.prob_same_width, self.prob_smaller_width])
                if choice == 0:
                    widths[i] = widths[i-1]
                elif choice == 1:
                    # widths[i] = np.random.uniform(self.min_body_width, widths[i-1])
                    # print(self.min_body_width, widths[i-1])
                    widths[i] = np.random.uniform(np.max([widths[i-1] - self.max_body_segment_reduction,
                                                          self.min_body_width]), widths[i-1])
                choices[i] = choice
                    
            heights[i] = np.random.uniform(self.min_body_height, self.max_body_height)
        
        for i in range(n_body_segments):
            vsk.rect(0, -0.5*heights[i], 0.5*widths[i], 0.5*heights[i], mode='radius')  # draw body part
            
            if use_filled_parts and np.random.uniform() < self.prob_fill:  # fill it
                fill_line_padding = np.random.uniform(self.min_fill_line_padding, self.max_fill_line_padding)
                if np.random.uniform() < 0.5:  # TODO: parametrize
                    n_fill = int(heights[i] / fill_line_padding)
                    for j in range(n_fill):
                        height_j = -heights[i] * j / n_fill
                        vsk.line(-0.5*widths[i], height_j, 0.5*widths[i], height_j)
                else:
                    n_fill = int(widths[i] / fill_line_padding)
                    for j in range(n_fill):
                        width_j = widths[i] * (j / n_fill - 0.5)
                        vsk.line(width_j, 0, width_j, -heights[i])
                    
            vsk.translate(0, -heights[i])
            
            if np.random.uniform() < self.prob_trapezoid and i < n_body_segments-1 and choices[i+1] == 1:  # add trapezoid
                trapezoid_height = np.random.uniform(self.min_trapezoid_height, self.max_trapezoid_height)
                vsk.polygon([(-0.5*widths[i], 0), (-0.5*widths[i+1], -trapezoid_height),
                             (0.5*widths[i+1], -trapezoid_height), (0.5*widths[i], 0)], close=True)
                vsk.translate(0, -trapezoid_height)
        
        self.tip_width = widths[-1]
    
    def draw_rocket_engine(self, vsk, inner_width, outer_width, height):
        vsk.polygon([(-0.5*outer_width, 0), (-0.5*inner_width, -height),
                             (0.5*inner_width, -height), (0.5*outer_width, 0)], close=True)
        
    def draw_rocket_bottom(self, vsk, width):
        # n_engines = np.random.randint(self.min_engines, self.max_engines + 1)
        n_engines = np.clip(np.random.geometric(p=self.prob_trial_engine), self.min_engines, self.max_engines)
        
        outer_padding = 0.05  # TODO: randomize
        inner_padding = 0.05  # TODO: incorporate this
        
        engine_width = (width - 2*outer_padding - (n_engines + 1)*inner_padding) / n_engines
        inner_engine_width = engine_width * np.random.uniform(self.min_inner_engine_width_gain, self.max_inner_engine_width_gain)  # TODO: randomize
        engine_height = np.random.uniform(self.min_engine_height_gain, self.max_engine_height_gain)/np.sqrt(n_engines)
        
        vsk.translate(-width / 2 + outer_padding + inner_padding + engine_width/2, 0)
        
        for i in range(n_engines):
            # TODO: draw up from 0
            self.draw_rocket_engine(vsk, inner_engine_width, engine_width, engine_height)
            if i < n_engines - 1:
                vsk.translate(engine_width + inner_padding, 0)
                
        vsk.translate(-width / 2 + outer_padding + inner_padding + engine_width/2, -engine_height)
        
    def draw_rocket(self, vsk):
        with vsk.pushMatrix():
            bottom_width = np.random.uniform(self.min_bottom_width, self.max_bottom_width)
            self.draw_rocket_bottom(vsk, bottom_width)
            self.draw_rocket_body(vsk, bottom_width)
            self.draw_rocket_tip(vsk, self.tip_width)  
            self.draw_rocket_extras(vsk)  

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.penWidth("0.4mm")
        
        # vsk.circle(0, 0, 0.001)
        # vsk.circle(0, 0, 1)
        
        vsk.scale(self.scale, self.scale)
        
        for y in range(self.n_y):
            with vsk.pushMatrix():
                for x in range(self.n_x):
                    self.draw_rocket(vsk)
                    
                    # vsk.circle(0, 0, 0.1)
                    
                    vsk.translate(self.grid_dist_x, 0)    
            vsk.translate(0, -self.grid_dist_y)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    RocketSketch.display()
