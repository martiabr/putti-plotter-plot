import vsketch
import numpy as np
from scipy.stats import norm

# What do we want to do here?
# - Just nice curves (with or without noise, with lines going between offsets?)
# - Random walk on predefined trajectory
# - Normal random walk
# - Grid of random walks

class curve:
    # def __init__(self, dt, vsk, mean, scale, gain, n):
    def __init__(self, dt, vsk):
        self.dt = dt
        self.vsk = vsk
        # self.mean = mean
        # self.scale = scale
        # self.gain = gain
        # self.n = n
    
    def get_point(self, t):
        # return np.array([np.sin(t), np.cos(t)])
        # return np.array([np.exp((np.sin(t)+0.2)**2), np.sin(t+0.5)*np.cos(t)])
        # return np.array([(np.sin(t)+0.2)**2, 0.3*np.sin(1.5*t+0.5)+1.2*np.sin(t+0.5)*np.cos(t-0.4)])
        
        # n = self.vsk.noise(1.0*t) - 0.5
        # r = 1.0 + 0.3*n
        # return np.array([r*np.sin(t), r*np.cos(t)])
        # return np.array([r*np.exp((np.sin(t)+0.2)**2), r*np.sin(t+0.5)*np.cos(t)])
        
        x = np.cos(t) + 1.0
        y = np.sin(t) + 1.0
        r = 1.0 + 1.3*np.interp(self.vsk.noise(x, y), (0, 1), (-1, 1))
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.array([x, y])
        
        # x = np.cos(t) + 1.0
        # y = np.sin(t) + 1.0
        # r = 1.0 
        # for i in range(self.n):
        #     r += self.gain[i] * (norm.pdf(t, self.mean[i]-2*np.pi, self.scale[i]) + norm.pdf(t, self.mean[i], self.scale[i]) + norm.pdf(t, 2*np.pi + self.mean[i], self.scale[i]))
        # x = r * np.cos(t)
        # y = r * np.sin(t)
        # return np.array([x, y])
    
    def get_line(self, t):
        return self.get_point(t), self.get_point(t + self.dt)

class WalkPathSketch(vsketch.SketchClass):
    
    dt = 0.01
    scale = vsketch.Param(1.0)
    
    n_x = vsketch.Param(4)
    n_y = vsketch.Param(6)
    
    grid_dist_x = vsketch.Param(4.6)
    grid_dist_y = vsketch.Param(4.6)
    
    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        N = int(2 * np.pi / self.dt)
        
        for y in range(self.n_y):
            with vsk.pushMatrix():
                for x in range(self.n_x):
                    vsk.noiseDetail(3, falloff=0.5)
                    vsk.noiseSeed(np.random.randint(1e6))
                    
                    c = curve(self.dt, vsk)
                    points = np.zeros((N,2))
                    for i in range(N):
                        t = self.dt * i
                        points[i] = c.get_point(t)
                        
                        # p_1, p_2  = c.get_line(t)
                        # vsk.line(self.scale*p_1[0], self.scale*p_1[1], self.scale*p_2[0], self.scale*p_2[1])
                    
                    vsk.noiseDetail(4)
                    
                    sub = vsketch.Vsketch()
                    
                    dt_walk = 0.5
                    N_walk = 1500
                    k_a = 0.03
                    k_t = 0.01
                    k_p = 0.2
                    k_v = 5.0
                    pos = np.zeros((N_walk,2))
                    pos[0] = points[0]
                    vel = np.zeros((N_walk,2))
                    acc = np.zeros((N_walk,2))
                    for i in range(1,N_walk):
                        t = dt_walk * i
                        pos_error = pos[i-1] - points[(i-1)%N]
                        acc[i] = k_a*(np.array([vsk.noise(k_t*t), vsk.noise(k_t*t+100000)])-0.5) - k_v * np.sign(vel[i-1]) * np.square(vel[i-1]) - k_p * np.sign(pos_error) * np.square(pos_error)
                        vel[i] = vel[i-1] + dt_walk * acc[i]
                        pos[i] = pos[i-1] + dt_walk * vel[i]
                        sub.line(self.scale*pos[i-1,0], self.scale*pos[i-1,1], self.scale*pos[i,0], self.scale*pos[i,1])
                    vsk.sketch(sub)
                    vsk.translate(self.grid_dist_x, 0)    
            vsk.translate(0, -self.grid_dist_y)
                
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
        
        # N = int(2 * np.pi / self.dt)
        # n = 10
        # mean = np.random.uniform(0.0, 2*np.pi, n)
        # scale = np.random.uniform(0.05, 0.5, n)
        # gain = np.random.uniform(0.01, 0.2, n)
        
        # c = curve(self.dt, vsk, mean, scale, gain, n)
        # c = curve(self.dt, vsk)
        
        # for i in range(N):
        #     t = self.dt * i
        #     p_1, p_2  = c.get_line(t)
        #     vsk.line(self.scale*p_1[0], self.scale*p_1[1], self.scale*p_2[0], self.scale*p_2[1])

        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    WalkPathSketch.display()
