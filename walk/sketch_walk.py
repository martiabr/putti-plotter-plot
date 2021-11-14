import vsketch
import numpy as np

class WalkSketch(vsketch.SketchClass):
    dt = vsketch.Param(1.0)
    N = vsketch.Param(200)
    
    n_x = vsketch.Param(6)
    n_y = vsketch.Param(9)
    
    grid_dist_x = vsketch.Param(3.0)
    grid_dist_y = vsketch.Param(3.0)
    
    scale = vsketch.Param(0.33)
    
    k_a = vsketch.Param(0.1)
    k_t = vsketch.Param(0.02)
    k_p = vsketch.Param(0.01)
    k_v = vsketch.Param(0.1)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        for y in range(self.n_y):
            with vsk.pushMatrix():
                for x in range(self.n_x):
                    pos = np.zeros((self.N,2))
                    vel = np.zeros((self.N,2))
                    acc = np.zeros((self.N,2))
                    sub = vsketch.Vsketch()
                    vsk.noiseSeed(np.random.randint(1e6))
                    for i in range(1,self.N):
                        t = self.dt * i
                        acc[i] = self.k_a*(np.array([vsk.noise(self.k_t*t), vsk.noise(self.k_t*t+100000)])-0.5) - self.k_v * np.sign(vel[i-1]) * np.square(vel[i-1]) - self.k_p * np.sign(pos[i-1]) * np.square(pos[i-1])# + 10 / (i-N) * (pos[i-1] - pos[0])
                        vel[i] = vel[i-1] + self.dt * acc[i]
                        pos[i] = pos[i-1] + self.dt * vel[i]
                        sub.line(self.scale*pos[i-1,0], self.scale*pos[i-1,1], self.scale*pos[i,0], self.scale*pos[i,1])
                    vsk.sketch(sub)
                    vsk.translate(self.grid_dist_x, 0)    
            vsk.translate(0, -self.grid_dist_y)
            
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    WalkSketch.display()
