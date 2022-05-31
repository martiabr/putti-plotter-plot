import vsketch
import numpy as np
import matplotlib.pyplot as plt

# - Somehow make path stay inside rectangle / the page instead of circle
#   - Maybe have to calculate shortest distance outside of rectangle?
# - Nonlinear transform for bounding radius instead of clipping - sigmoid?
# - Deal with ends - go to zero (x)
# - Nonlinear ends
# - Randomize starting position of circle stroke to avoid weird artifact?
# - Max acceleration norm (x)
# - Max velocity norm (x)
# - Somehow get the line to go back to start?
# - Randomize parameters for different colors to spice things up?

class WalkColorSketch(vsketch.SketchClass):
    dt = vsketch.Param(0.25)
    N = vsketch.Param(1250)
    
    n_x = vsketch.Param(1)
    n_y = vsketch.Param(1)
    
    n_colors = vsketch.Param(3)
    
    grid_dist_x = vsketch.Param(3.0)
    grid_dist_y = vsketch.Param(3.0)
    
    scale = vsketch.Param(2.5)
    
    k_acc = vsketch.Param(0.1)
    freq_acc_noise = vsketch.Param(0.02)
    
    k_p = vsketch.Param(0.01, decimals=4)
    k_v = vsketch.Param(0.1)
    k_dv = vsketch.Param(0.0)
    
    acc_max = vsketch.Param(0.020)
    vel_max = vsketch.Param(0.20)
    
    x_0_max = vsketch.Param(2.0)
    y_0_max = vsketch.Param(2.0)
    
    path_mode = vsketch.Param("LINE", choices=["LINE", "CIRCLES"])
    
    pos_penalty_mode = vsketch.Param("CIRCLE", choices=["CIRCLE", "RECT"])
    
    center_radius = vsketch.Param(0.2)
    freq_radius = vsketch.Param(0.5)
    amplitude_radius = vsketch.Param(1.0)
    min_radius = vsketch.Param(0.01)
    max_radius = vsketch.Param(1.0)
    
    detail = vsketch.Param(0.015)
    
    occult = vsketch.Param(False)
    
    ends_go_to_zero = vsketch.Param(False)
    end_start = vsketch.Param(100)
    
    width = 21.0
    height = 29.7

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        
        # np.random.seed(1234)
        
        with vsk.pushMatrix():
            for y in range(self.n_y):
                with vsk.pushMatrix():
                    for x in range(self.n_x):
                        for i in range(self.n_colors):
                            pos = np.zeros((self.N,2))
                            vel = np.zeros((self.N,2))
                            acc = np.zeros((self.N,2))
                            pos[0] = np.random.uniform(low=[-self.x_0_max, -self.y_0_max], high=[self.x_0_max, self.y_0_max])
                            
                            sub = vsketch.Vsketch()
                            sub.stroke(i+1)
                            sub.detail(str(self.detail) + "mm")
                            # sub.penWidth("0.01mm")
                            # sub.strokeWeight(1)
                            vsk.noiseSeed(np.random.randint(1e6))
                            
                            for i in range(1,self.N):
                                t = self.dt * i
                                
                                acc_noise = np.array([vsk.noise(self.freq_acc_noise*t), vsk.noise(self.freq_acc_noise*t + 1e5)]) - 0.5
                                acc[i] = self.k_acc * acc_noise
                                acc[i] -= self.k_v * np.sign(vel[i-1]) * np.square(vel[i-1])
                                acc[i] -= self.k_dv / self.dt**2 * np.sign(vel[i-1] - vel[i-2]) * np.square(vel[i-1] - vel[i-2])
                                if self.pos_penalty_mode == "CIRCLE":
                                    acc[i] -= self.k_p * np.sign(pos[i-1]) * np.square(pos[i-1])
                                elif self.pos_penalty_mode == "RECT":
                                    if np.abs(pos[i-1,0]) > self.width / 2 or np.abs(pos[i-1,1]) > self.height / 2:
                                        acc[i] -= self.k_p * np.sign(pos[i-1]) * np.square(pos[i-1])
                                
                                acc_norm = np.linalg.norm(acc[i])
                                if acc_norm > self.acc_max:
                                    acc[i] *= self.acc_max / acc_norm
                                    
                                vel[i] = vel[i-1] + self.dt * acc[i]
                                vel_norm = np.linalg.norm(vel[i])
                                if vel_norm > self.vel_max:
                                    vel[i] *= self.vel_max / vel_norm
                                    
                                pos[i] = pos[i-1] + self.dt * vel[i]
                                
                                if self.path_mode == "CIRCLES":
                                    radius = self.center_radius * (1.0 + self.amplitude_radius * (vsk.noise(self.freq_radius * t + 1e2) - 0.5))
                                    
                                    if self.ends_go_to_zero:
                                        if self.N - i < self.end_start:
                                            radius *= ((self.N - i) / self.end_start)**0.5
                                        elif i < self.end_start:
                                            radius *= (i / self.end_start)**0.5
                                            
                                    radius = np.clip(radius, self.min_radius, self.max_radius)
                                    sub.circle(self.scale*pos[i,0], self.scale*pos[i,1], radius=radius)
                                elif self.path_mode == "LINE":
                                    sub.line(self.scale*pos[i-1,0], self.scale*pos[i-1,1], self.scale*pos[i,0], self.scale*pos[i,1])
                            vsk.sketch(sub)
                        vsk.translate(self.grid_dist_x, 0)    
                vsk.translate(0, -self.grid_dist_y)
                
        if self.occult:
            vsk.vpype("occult -i")
            
                    
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    WalkColorSketch.display()
