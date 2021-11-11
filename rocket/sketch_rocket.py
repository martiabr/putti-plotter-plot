from numpy.core.numeric import outer
import vsketch
import numpy as np
from enum import Enum

# - random part that goes in on both sides
# - fins (either just triangle or parallelogram or trapezoid) (either at the bottom or somewhere higher up)
# - side boosters (could also have 4 boosters, so also one in front)
# - star ship easter egg?
# - improve pointy tip

class RocketSketch(vsketch.SketchClass):
    
    ### Main params:
    
    n_x = vsketch.Param(18)
    n_y = vsketch.Param(6)
    grid_dist_x = vsketch.Param(4.0)
    grid_dist_y = vsketch.Param(19.0)
    scale = vsketch.Param(0.22, decimals=2)
    
    ### Engine params:
    
    min_engines = vsketch.Param(1)
    max_engines = vsketch.Param(5)
    
    engine_types = Enum('EngineType', 'triangle trapezoid dome')
    prob_engine_types = [0.7, 0.15, 0.15]
    
    prob_trial_engine = vsketch.Param(0.5)  # p in geometric distribution
    
    min_inner_engine_width_gain = vsketch.Param(0.2)
    max_inner_engine_width_gain = vsketch.Param(0.6)
    
    min_engine_height_gain = vsketch.Param(0.5)
    max_engine_height_gain = vsketch.Param(1.2)
    
    min_bottom_width = vsketch.Param(1.0)
    max_bottom_width = vsketch.Param(2.0)
    
    min_trapezoid_engine_height_gain = vsketch.Param(0.5)
    max_trapezoid_engine_height_gain = vsketch.Param(1.2)
    
    min_trapezoid_engine_width_gain = vsketch.Param(0.7)
    max_trapezoid_engine_width_gain = vsketch.Param(1.2)

    min_trapezoid_engine_height_trapezoid_gain = vsketch.Param(0.16)
    max_trapezoid_engine_height_trapezoid_gain = vsketch.Param(0.4)
    
    min_dome_engine_angle = vsketch.Param(0.1)
    max_dome_engine_angle = vsketch.Param(1.0)
    
    min_dome_engine_height_gain = vsketch.Param(1.0)
    max_dome_engine_height_gain = vsketch.Param(2.5)
    
    min_dome_engine_inner_width_gain = vsketch.Param(0.2)
    max_dome_engine_inner_width_gain = vsketch.Param(0.4)
    
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
    
    ### Fins:
    
    prob_fins = vsketch.Param(0.2)
    fin_types = Enum('FinType', 'delta clipped_parallelogram clipped_delta')
    prob_fin_types = [0.3, 0.3, 0.4]
    
    min_fin_width = vsketch.Param(0.3)
    max_fin_width = vsketch.Param(1.0)
    
    ### Details:
    
    prob_details = vsketch.Param(0.2)
    detail_types = Enum('DetailType', 'circle rect triangle rect_outside')
    prob_detail_types = [0.25, 0.3, 0.25, 0.2]
    
    min_detail_circle_radius_gain = vsketch.Param(0.1)
    max_detail_circle_radius_gain = vsketch.Param(0.5)
    
    min_detail_rect_width_gain = vsketch.Param(0.1)
    max_detail_rect_width_gain = vsketch.Param(0.3)
    min_detail_rect_height_gain = vsketch.Param(0.25)
    max_detail_rect_height_gain = vsketch.Param(0.75)
    
    min_detail_triangle_width_gain = vsketch.Param(0.05)
    max_detail_triangle_width_gain = vsketch.Param(0.15)
    
    min_detail_rect_outside_width = vsketch.Param(0.1)
    max_detail_rect_outside_width = vsketch.Param(0.4)
    
    min_detail_rect_outside_height = vsketch.Param(0.3)
    max_detail_rect_outside_height = vsketch.Param(0.8)
    
    ### Tip params:
    
    tip_types = Enum('TipType', 'ellipse_pointy triangle trapezoid_fancy ellipse')
    prob_tips = [0.21, 0.21, 0.21, 0.37]
    
    min_tip_height = vsketch.Param(1.25)
    max_tip_height = vsketch.Param(2.5)
    
    min_trapezoid_tip_gain = vsketch.Param(0.1)
    max_trapezoid_tip_gain = vsketch.Param(0.2)
    
    consider_ellipse_width = vsketch.Param(0.8)
    
    min_tip_ellipse_line_height_gain = vsketch.Param(1.5)
    max_tip_ellipse_line_height_gain = vsketch.Param(3.0)
    
    ###
    
    def draw_rocket_tip_ellipse_pointy(self, vsk, width):
        angle = np.random.uniform(0, np.pi / 2)  # TODO: parametrize
        height = np.random.uniform(self.min_tip_height, self.max_tip_height)
        a = 0.5 * width / (1 - np.cos(angle))
        b = 0.5 * height / np.sin(angle)
        x = a*np.cos(angle)
        vsk.arc(- x, 0, 2 * a, 2 * b, 0, angle)
        vsk.arc(x, 0, 2 * a, 2 * b, np.pi - angle, np.pi)
        
    def draw_rocket_tip_ellipse(self, vsk, width):
        # Trapezoid:
        trapezoid_width_gain = np.random.uniform(1.5, 1.8)  # TODO
        trapezoid_height = np.random.uniform(0.4, 0.6)  # TODO
        upper_width = trapezoid_width_gain * width
        vsk.polygon([(-0.5*width, 0), (-0.5*upper_width, -trapezoid_height),
                        (0.5*upper_width, -trapezoid_height), (0.5*width, 0)], close=True)
        vsk.translate(0, -trapezoid_height)
        
        # Lines up:
        line_height_gain = np.random.uniform(self.min_tip_ellipse_line_height_gain, self.max_tip_ellipse_line_height_gain)
        line_height = line_height_gain * width
        vsk.line(-0.5*upper_width, 0, -0.5*upper_width, -line_height)
        vsk.line(0.5*upper_width, 0, 0.5*upper_width, -line_height)
        vsk.translate(0, -line_height)
        
        # Ellipse arc:
        height = np.random.uniform(self.min_tip_height, self.max_tip_height)
        vsk.arc(0, 0, upper_width, height, 0, np.pi)
    
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
        if width < self.consider_ellipse_width:
            choice = np.random.choice(len(self.tip_types), p=self.prob_tips)
        else:
            choice = np.random.choice(len(self.tip_types)-1, p=self.prob_tips[:3]/np.sum(self.prob_tips[:3]))
        if choice == self.tip_types.ellipse_pointy.value - 1:
            self.draw_rocket_tip_ellipse_pointy(vsk, width)
        elif choice == self.tip_types.triangle.value - 1:
            self.draw_rocket_tip_triangle(vsk, width)
        elif choice == self.tip_types.trapezoid_fancy.value - 1:
            self.draw_rocket_tip_trapezoid_fancy(vsk, width)
        elif choice == self.tip_types.ellipse.value - 1:
            self.draw_rocket_tip_ellipse(vsk, width)
    
    ###############################################################################
    
    def draw_rocket_extras(self, vsk):
        pass
    
    ###############################################################################
    
    def draw_rocket_fins_delta(self, vsk, width, height):
        fin_width = np.random.uniform(self.min_fin_width, self.max_fin_width)
        vsk.triangle(-0.5*width, 0, -0.5*width - fin_width, 0, -0.5*width, -height)
        vsk.triangle(0.5*width, 0, 0.5*width + fin_width, 0, 0.5*width, -height)
        
    def draw_rocket_fins_clipped_delta(self, vsk, width, height):
        fin_width = np.random.uniform(self.min_fin_width, self.max_fin_width)
        fin_height_1 = 0.4 * height  # TODO
        vsk.polygon([(-0.5*width, 0), (-0.5*width - fin_width, 0), (-0.5*width - fin_width, -fin_height_1), (-0.5*width, -height)], close=True)
        vsk.polygon([(0.5*width, 0), (0.5*width + fin_width, 0), (0.5*width + fin_width, -fin_height_1), (0.5*width, -height)], close=True)
    
    def draw_rocket_fins_clipped_parallelogram(self, vsk, width, height):
        fin_width = np.random.uniform(self.min_fin_width, self.max_fin_width)
        fin_height_1 = 0.4 * height  # TODO
        fin_height_2 = -0.4 * height  # TODO
        vsk.polygon([(-0.5*width, 0), (-0.5*width - fin_width, -fin_height_2), (-0.5*width - fin_width, -fin_height_1), (-0.5*width, -height)], close=True)
        vsk.polygon([(0.5*width, 0), (0.5*width + fin_width, -fin_height_2), (0.5*width + fin_width, -fin_height_1), (0.5*width, -height)], close=True)
    
    def draw_rocket_fins(self, vsk, width, height):
        choice = np.random.choice(len(self.fin_types), p=self.prob_fin_types)
        if choice == self.fin_types.delta.value - 1:
            self.draw_rocket_fins_delta(vsk, width, height)
        elif choice == self.fin_types.clipped_parallelogram.value - 1:
            self.draw_rocket_fins_clipped_parallelogram(vsk, width, height)
        elif choice == self.fin_types.clipped_delta.value - 1:
            self.draw_rocket_fins_clipped_delta(vsk, width, height)
    
    ###############################################################################
    
    def draw_rocket_detail_circle(self, vsk, height, width):
        max_radius = np.min([height, width])
        radius = np.random.uniform(self.min_detail_circle_radius_gain*max_radius, self.max_detail_circle_radius_gain*max_radius)
        vsk.circle(0, -0.5*height, radius)
        
    def draw_rocket_detail_rect(self, vsk, height, width):
        gain_w = np.random.uniform(self.min_detail_rect_width_gain, self.max_detail_rect_width_gain)
        gain_h_1 = np.random.uniform(self.min_detail_rect_height_gain, self.max_detail_rect_height_gain)
        gain_h_2 = np.random.uniform(self.min_detail_rect_height_gain, self.max_detail_rect_height_gain)
        vsk.polygon([(-gain_w*width, -gain_h_1*height), (-gain_w*width, -gain_h_2*height), (gain_w*width, -gain_h_2*height), (gain_w*width, -gain_h_1*height)], close=True)  # TODO: randomize

    def draw_rocket_detail_triangle(self, vsk, height, width):
        triangle_width_gain = np.random.uniform(self.min_detail_triangle_width_gain, self.max_detail_triangle_width_gain)
        vsk.triangle(-triangle_width_gain*width, 0, triangle_width_gain*width, 0, 0, -2*triangle_width_gain*width)
    
    def draw_rocket_detail_rect_outside(self, vsk, height, width):
        rect_width = np.random.uniform(self.min_detail_rect_outside_width, self.max_detail_rect_outside_width)
        rect_height = np.min([height, np.random.uniform(self.min_detail_rect_outside_height, self.max_detail_rect_outside_height)])
        
        vsk.rect(-0.5*(width + rect_width), -0.5*height, rect_width, rect_height, mode='center')
        vsk.rect(0.5*(width + rect_width), -0.5*height, rect_width, rect_height, mode='center')
    
    ###############################################################################
    
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
            
            do_fill = (use_filled_parts and np.random.uniform() < self.prob_fill)
            do_fins = (i == 0 and np.random.uniform() < self.prob_fins)
            do_details = (not do_fill and not do_fins and np.random.uniform() < self.prob_details)
            
            if do_fill:  # fill it
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
            elif do_details:  # add random details
                detail_choice = np.random.choice(len(self.detail_types), p=self.prob_detail_types)
                if detail_choice == self.detail_types.circle.value - 1:
                    self.draw_rocket_detail_circle(vsk, heights[i], widths[i])
                elif detail_choice == self.detail_types.rect.value - 1:
                    self.draw_rocket_detail_rect(vsk, heights[i], widths[i])
                elif detail_choice == self.detail_types.triangle.value - 1:
                    self.draw_rocket_detail_rect_outside(vsk, heights[i], widths[i])
            
            if do_fins:
                self.draw_rocket_fins(vsk, widths[i], heights[i])
                    
            vsk.translate(0, -heights[i])
            
            if np.random.uniform() < self.prob_trapezoid and i < n_body_segments-1 and choices[i+1] == 1:  # add trapezoid
                trapezoid_height = np.random.uniform(self.min_trapezoid_height, self.max_trapezoid_height)
                vsk.polygon([(-0.5*widths[i], 0), (-0.5*widths[i+1], -trapezoid_height),
                             (0.5*widths[i+1], -trapezoid_height), (0.5*widths[i], 0)], close=True)
                vsk.translate(0, -trapezoid_height)
        
        self.tip_width = widths[-1]
    
    ###############################################################################
    def draw_rocket_engine_triangle(self, vsk, inner_width, outer_width, height):
        vsk.polygon([(-0.5*outer_width, 0), (-0.5*inner_width, -height),
                     (0.5*inner_width, -height), (0.5*outer_width, 0)], close=True)
    
    def draw_rocket_engine_triangles(self, vsk, width):
        n_engines = np.clip(np.random.geometric(p=self.prob_trial_engine), self.min_engines, self.max_engines)
    
        outer_padding = 0.05  # TODO: randomize
        inner_padding = 0.05  # TODO: randomize 
        
        engine_width = (width - 2*outer_padding - (n_engines + 1)*inner_padding) / n_engines
        inner_engine_width = engine_width * np.random.uniform(self.min_inner_engine_width_gain, self.max_inner_engine_width_gain)  # TODO: randomize
        engine_height = np.random.uniform(self.min_engine_height_gain, self.max_engine_height_gain)/np.sqrt(n_engines)
        
        vsk.translate(-width / 2 + outer_padding + inner_padding + engine_width/2, 0)
        
        for i in range(n_engines):
            # TODO: draw up from 0
            self.draw_rocket_engine_triangle(vsk, inner_engine_width, engine_width, engine_height)
            if i < n_engines - 1:
                vsk.translate(engine_width + inner_padding, 0)
                
        vsk.translate(-width / 2 + outer_padding + inner_padding + engine_width/2, -engine_height)
        
    def draw_rocket_engine_trapezoid(self, vsk, width):
        inner_engine_width = width * np.random.uniform(self.min_inner_engine_width_gain, self.max_inner_engine_width_gain)
        engine_height = width * np.random.uniform(self.min_trapezoid_engine_height_gain, self.max_trapezoid_engine_height_gain)
        outer_engine_width = np.max([1.4*inner_engine_width, width * np.random.uniform(self.min_trapezoid_engine_width_gain,
                                                                                       self.max_trapezoid_engine_width_gain)])
        vsk.polygon([(-0.5*outer_engine_width, 0), (-0.5*inner_engine_width, -engine_height),
                (0.5*inner_engine_width, -engine_height), (0.5*outer_engine_width, 0)], close=True)
        
        vsk.translate(0, -engine_height)
        
        trapezoid_height = width * np.random.uniform(self.min_trapezoid_engine_height_trapezoid_gain, self.max_trapezoid_engine_height_trapezoid_gain)
        vsk.polygon([(-0.5*inner_engine_width, 0), (-0.5*width, -trapezoid_height),
                        (0.5*width, -trapezoid_height), (0.5*inner_engine_width, 0)], close=True)
        
        vsk.translate(0, -trapezoid_height)
    
    def draw_rocket_engine_dome(self, vsk, width):
        angle = np.random.uniform(self.min_dome_engine_angle, self.max_dome_engine_angle)
        height = width * np.random.uniform(self.min_dome_engine_height_gain, self.max_dome_engine_height_gain)
        inner_engine_width = width * np.random.uniform(self.min_dome_engine_inner_width_gain, self.max_dome_engine_inner_width_gain)
        
        a = 0.5 * (width-inner_engine_width) / (1 - np.cos(angle))
        b = 0.5 * height / np.sin(angle)
        x = a * np.cos(angle)
        y = b * np.sin(angle)
        
        vsk.arc(- x + 0.5*inner_engine_width, 0, 2 * a, 2 * b, 0, angle)
        vsk.arc(x - 0.5*inner_engine_width, 0, 2 * a, 2 * b, np.pi - angle, np.pi)
        
        vsk.line(-0.5*width, 0, 0.5*width, 0)
        
        vsk.translate(0, -y)
        
    def draw_rocket_bottom(self, vsk, width):
        engine_choice = np.random.choice(len(self.engine_types), p=self.prob_engine_types)
        if engine_choice == self.engine_types.triangle.value - 1:
            self.draw_rocket_engine_triangles(vsk, width)
        elif engine_choice == self.engine_types.trapezoid.value - 1:
            self.draw_rocket_engine_trapezoid(vsk, width)
        elif engine_choice == self.engine_types.dome.value - 1:
            self.draw_rocket_engine_dome(vsk, width)
        
    ###############################################################################
        
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
