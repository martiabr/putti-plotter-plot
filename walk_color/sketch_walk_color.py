import vsketch
import numpy as np
import matplotlib.pyplot as plt

# What is the plan here?
# - Initially no grid, just a single doodle
# - But several doodles layered on top of each other with different colorss
# - From there what can be done to make it cooler?
# - Try with grid again
# - Tune parameters and try out different parametrizations
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
    k_t = vsketch.Param(0.02)
    k_p = vsketch.Param(0.01)
    k_v = vsketch.Param(0.1)
    k_dv = vsketch.Param(0.0)
    
    x_0_max = vsketch.Param(2.0)
    y_0_max = vsketch.Param(2.0)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
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
                            vsk.noiseSeed(np.random.randint(1e6))
                            for i in range(1,self.N):
                                t = self.dt * i
                                acc[i] = self.k_acc*(np.array([vsk.noise(self.k_t*t),
                                                             vsk.noise(self.k_t*t+100000)])-0.5) - \
                                                                self.k_v * np.sign(vel[i-1]) * np.square(vel[i-1]) - \
                                                                self.k_p * np.sign(pos[i-1]) * np.square(pos[i-1]) - \
                                                                self.k_dv / self.dt**2 * np.sign(vel[i-1] - vel[i-2]) * np.square(vel[i-1] - vel[i-2]) \
                                                                # - 100 / (i-self.n_x + 1e-2)**2 * (pos[i-1] - pos[0]) \
                                                                # + 10 / (i-self.n_x + 1e-2)**2 * (vel[i-1] + vel[0])
                                vel[i] = vel[i-1] + self.dt * acc[i]
                                pos[i] = pos[i-1] + self.dt * vel[i]
                                sub.line(self.scale*pos[i-1,0], self.scale*pos[i-1,1], self.scale*pos[i,0], self.scale*pos[i,1])
                            vsk.sketch(sub)
                        vsk.translate(self.grid_dist_x, 0)    
                vsk.translate(0, -self.grid_dist_y)
                    
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    WalkColorSketch.display()
