import vsketch
import numpy as np

# What do we want to do here?
# - Just nice curves (with or without noise, with lines going between offsets?)
# - Random walk on predefined trajectory
# - Normal random walk
# - Grid of random walks

class curve:
    def __init__(self, dt, vsk):
        self.dt = dt
        self.vsk = vsk
    
    def get_point(self, t):
        # return np.array([np.sin(t), np.cos(t)])
        # return np.array([np.exp((np.sin(t)+0.2)**2), np.sin(t+0.5)*np.cos(t)])
        # return np.array([(np.sin(t)+0.2)**2, 0.3*np.sin(1.5*t+0.5)+1.2*np.sin(t+0.5)*np.cos(t-0.4)])
        
        n = self.vsk.noise(1.0*t) - 0.5
        r = 1.0 + 0.3*n
        # return np.array([r*np.sin(t), r*np.cos(t)])
        return np.array([r*np.exp((np.sin(t)+0.2)**2), r*np.sin(t+0.5)*np.cos(t)])
    
    def get_line(self, t):
        return self.get_point(t), self.get_point(t+self.dt)

class WalkSketch(vsketch.SketchClass):
    # Sketch parameters:
    # radius = vsketch.Param(2.0)
    
    dt = 0.5
    scale = 3

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        N = 400
        k_a = 0.1
        k_t = 0.02
        k_p = 0.01
        k_v = 0.1
        pos = np.zeros((N,2))
        vel = np.zeros((N,2))
        acc = np.zeros((N,2))
        # jerk = np.zeros((N,2))
        for i in range(1,N):
            t = self.dt * i
            # jerk[i] = 0.1*np.random.normal(scale=1, size=2) - 1 * np.sign(acc[i-1]) * acc[i-1].T @ acc[i-1] - 1 * np.sign(vel[i-1]) * vel[i-1].T @ vel[i-1] - 1 * np.sign(pos[i-1]) * pos[i-1].T @ pos[i-1]
            # acc[i] = acc[i-1] + self.dt * jerk[i]
            acc[i] = k_a*(np.array([vsk.noise(k_t*t), vsk.noise(k_t*t+100000)])-0.5) - k_v * np.sign(vel[i-1]) * np.square(vel[i-1]) - k_p * np.sign(pos[i-1]) * np.square(pos[i-1])# + 10 / (i-N) * (pos[i-1] - pos[0])
            vel[i] = vel[i-1] + self.dt * acc[i]
            # vel[i] = k_a * np.array([vsk.noise(k_t*t), vsk.noise(k_t*t+100000)])-0.5
            pos[i] = pos[i-1] + self.dt * vel[i]
            # pos[i] = np.array([vsk.noise(t), vsk.noise(t+100000)])-0.5
            vsk.line(self.scale*pos[i-1,0], self.scale*pos[i-1,1], self.scale*pos[i,0], self.scale*pos[i,1])
            
        # t = np.array([self.dt * i for i in range(N)])
        # plt.figure()
        # plt.plot(t, pos)
        # plt.figure()
        # plt.plot(t, vel)
        # plt.figure()
        # plt.plot(t, acc)
        # plt.figure()
        # plt.plot(t, jerk)
        # plt.show()
        
        # c = curve(self.dt, vsk)
        
        # N = 2000
        # for i in range(N):
        #     t = self.dt * i
        #     p_1, p_2  = c.get_line(t)
        #     vsk.line(self.scale*p_1[0], self.scale*p_1[1], self.scale*p_2[0], self.scale*p_2[1])

        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    WalkSketch.display()
