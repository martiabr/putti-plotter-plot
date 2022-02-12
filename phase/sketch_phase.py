import vsketch
import numpy as np
import random

# Initial idea:
# Flow fields, phase plot, attractors, perlin noise, nonlinear time-varying dynamics
# 
# Want to be able to either visualize grid of derivatives of draw phase plane trajectories (integrate)
# The underlying dynamics should be seperated from the flow field viz,
# such that it should be easy to change out perlin noise based flow field for nonlinear chaos system or combination.
# Scipy should be fine for integrating? Or just RK4.

# TODO:
# - randomized particle positions (x)
# - draw grid and border (x)
# - integrate backwards as well (x)
# - express system in polar coordinates
# - heuristic for picking good particle positions
# - better control of scaling and center 
# - menu of different fs
# - menu of different styles of grid
# - menu of different styles of paths
# - customize noise
# - Add custom parameters a la scipy
# - Fix function mess
# - Draw flow field function
# - Some sort of normalization of derivatives (like perlin flow field) such that the path is traced out better/more efficiently? 
# - Perfect border using some math
# - Grid of nice smooth perlin noise boxes, grid of sine type boxes


def draw_border(vsk, x_bounds, y_bounds):
    vsk.line(x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[0])
    vsk.line(x_bounds[0], y_bounds[1], x_bounds[1], y_bounds[1])
    vsk.line(x_bounds[0], y_bounds[0], x_bounds[0], y_bounds[1])
    vsk.line(x_bounds[1], y_bounds[0], x_bounds[1], y_bounds[1])


def integrate_RK4(x, f, t, dt):
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + dt * k1 / 2.0)
    k3 = f(t + 0.5 * dt, x + dt * k2 / 2.0)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        
def trace_trajectory(vsk, f, traj_length, x_0, y_0, dt, x_bounds=None, y_bounds=None):
    z = np.zeros((traj_length+1,2))
    z[0] = np.array([x_0, y_0])
    for i in range(traj_length):
        t = dt * i
        z[i+1] = integrate_RK4(z[i], f, t, dt)
                
        if x_bounds is not None and (z[i+1,0] < x_bounds[0] or z[i+1,0] > x_bounds[1]):
            break
        if y_bounds is not None and (z[i+1,1] < y_bounds[0] or z[i+1,1] > y_bounds[1]):
            break
        
        vsk.line(z[i,0], z[i,1], z[i+1,0], z[i+1,1])

class PhaseSketch(vsketch.SketchClass):
    
    border = vsketch.Param(True)
    grid = vsketch.Param(False)
    backward = vsketch.Param(True)
    randomize = vsketch.Param(True)
    field = vsketch.Param(False)
    particles = vsketch.Param(True)
    
    x_grid = vsketch.Param(1)
    y_grid = vsketch.Param(1)
    
    N_particles_x = vsketch.Param(20)
    N_particles_y = vsketch.Param(20)
    N_traj_steps = vsketch.Param(50)
    dt = vsketch.Param(2e-2)
    scale = vsketch.Param(0.5)
    x_dist = vsketch.Param(10.0)
    y_dist = vsketch.Param(10.0)
    
    x_min = vsketch.Param(4.0)
    x_max = vsketch.Param(4.0)
    y_min = vsketch.Param(4.0)
    y_max = vsketch.Param(4.0)
    
    norm_max = vsketch.Param(0.2)
    
    f_n = vsketch.Param(0.5)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        
        def f_perlin(t, z):
            x, y = z
            angle = 2 * np.pi * vsk.noise(x_bounds[0] + self.f_n*x, y_bounds[0] + self.f_n*y)
            dx = np.cos(angle)
            dy = np.sin(angle)
            return np.array([dx, dy])
        
        def f_pendulum(t, z):
            dx = z[1]
            dy = -1.0 * np.sin(z[0]) - 1.0 * z[1]
            return np.array([dx, dy])
        
        def f_experiment(t, z):
            # dx = z[1]
            # dy = - z[0] + 0.5*z[1]
            # dy = 0.05*z[1] + np.cos(z[0] + z[1])**2
            # dx = -2.0 * z[0] - 2 * z[1] * np.abs(1 - z[1])
            # dy = 1.0 * z[0] * np.abs(1 - z[0]) + 1.0 * z[1]
            
            x, y = z
            
            dx = y
            dy = - 0.1 * (x - 1)**3 - 0.4 * np.abs(np.sin(y))
            
            # dx = x + y + x**2 + y**2
            # dy = x - y - x**2 + y**2
            
            # dx = x + 3*y + x**2 + y**2
            # dy = x - y - 2*x**2 + y**2
            
            # a = 1.0
            # b = 1.0
            # dx = -b*y**2 - 0*x**2 + np.cos(a*y)
            # dy = -b*x**2 + y**2 + np.cos(a*y)
            
            
            # a = 1.0
            # b = 1.0
            # c = 0.2
            # g = 0.0
            # w = 0.5
            # dx = y
            # dy = - a * y - b * x - c * x**3 + g * np.cos(w*t)
            
            # Van der Pool
            # dx = y
            # dy = -x + 0.4 * (1 - x**2) * y
            
            return np.array([dx, dy])
        
        def f_1(t, z):
            x, y = z
            dx = 1.0 * np.sin(0.5*x) + np.sin(1.0*y)
            dy = 1.0 * np.sin(1.5*x) - np.sin(0.5*y)
            return np.array([dx, dy])
        
        def f_2(t, z):
            x, y = z
            dx = - 0.5 * np.sin(0.4*x) + np.sin(1.0*y)
            dy = 1.0 * np.sin(1.6*x) - 1.5 * np.sin(0.1*y)
            return np.array([dx, dy])
        
        def f_3(t, z):
            x, y = z
            dx = - 0.5 * np.cos(0.4*x) + np.sin(1.0*y) - 0.3*y
            dy = 1.0 * np.sin(1.6*x) + 0.5 * np.cos(0.2*y)
            return np.array([dx, dy])
        
        def f_4(t, z):
            x, y = z
            dx = np.cos(0.5*x) + np.sin(y)
            dy = np.sin(1.6*x) + 0.6*np.cos(0.2*y)
            return np.array([dx, dy])
        
        def f_5(t, z):
            x, y = z
            dx = - 0.5 * np.cos(-1.0*x) + 1.5*np.sin(1.0*y)
            dy = -1.5 * np.sin(0.5*x) - y * 0.5 * np.cos(0.5*y)
            return np.array([dx, dy])
        
        def f_6(t, z):
            x, y = z
            dx = - 0.5 * np.cos(-1.0*x+1) + 1.5*np.sin(1.0*y)
            dy = np.cos(0.5*x-1) - y * 0.5 * np.cos(0.5*y)
            return np.array([dx, dy])
        
        def f_7(t, z):
            x, y = z
            dx = -1.5 * np.cos(-0.5*x) - 1.5*np.sin(1.0*y)
            dy = np.cos(0.5*x-1) - 1.5 * np.cos(0.5*y)
            return np.array([dx, dy])
        
        def f_8(t, z):
            x, y = z
            dx = - 0.5 * np.sin(0.4*x) - np.cos(1.0*y)
            dy = -1.0 * np.sin(0.4*x) - 1.5 * np.cos(0.4*y)
            return np.array([dx, dy])
        
        def f_9(t, z):
            x, y = z
            dx = np.cos(0.5*x) - np.sin(y)
            dy = 0.4*np.sin(1.6*x) + 0.6*np.cos(0.2*y)
            return np.array([dx, dy])
        
        def f_10(t, z):
            x, y = z
            dx = -0.4*np.cos(0.5*x) + 0.5*np.sin(y)
            dy = -0.4*np.sin(-0.5*x) + 0.5*np.sin(0.5*y)
            return np.array([dx, dy])
        
        def f_11(t, z):
            x, y = z
            dx = - 0.5 * np.cos(-0.7*x) + np.sin(1.0*y) - 0.2*y
            dy = 0.6 * np.sin(0.5*x) + 0.3 * np.cos(0.4*y)
            return np.array([dx, dy])
        
        def f_12(t, z):
            x, y = z
            dx = -0.5 * np.cos(0.5*x) + np.sin(0.5*y)
            dy = - np.sin(0.5*x) - 0.1*np.cos(0.2*y) * x * y - 0.3 * y
            return np.array([dx, dy])
        
        def f_13(t, z):
            x, y = z
            dx = - 0.5 * np.cos(-1.0*x+0.6) - 1.2*np.sin(1.2*y - 0.5)
            dy = np.cos(0.5*x-1) + x * 0.2 * np.cos(0.5*y - 0.3)
            return np.array([dx, dy])
        
        def f_14(t, z):
            x, y = z
            dx = y + 0.4*x*np.sin(x)
            dy = 0.06 * (1 - x**2) * y + 0.5*np.cos(0.2*x + 0.2)**2
            return np.array([dx, dy])
        
        def f_15(t, z):
            x, y = z
            dx = 0.3 * np.cos(-1.0*x+1) - 0.2*np.cos(1.2*y)
            dy = - 0.5 * np.cos(0.3*x-1) - y * 0.3 * np.cos(-0.2*y)
            return np.array([dx, dy])
        
        f_swirls = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11, f_12, f_13, f_14, f_15]
        random.shuffle(f_swirls)
            
        vsk.scale(self.scale, self.scale)
        x_bounds = np.array([-self.x_min, self.x_max])
        y_bounds = np.array([-self.y_min, self.y_max])
        
        if self.randomize:
            xy_points = np.random.uniform(low=[x_bounds[0], y_bounds[0]], high=[x_bounds[1], y_bounds[1]], size=(self.N_particles_x,2))
        else:
            x_range = np.linspace(x_bounds[0], x_bounds[1], self.N_particles_x)
            y_range = np.linspace(y_bounds[0], y_bounds[1], self.N_particles_y)
            X, Y = np.meshgrid(x_range, y_range)
            xy_points = np.vstack((X.flatten(), Y.flatten())).T

        for i in range(self.y_grid):
            with vsk.pushMatrix():
                for j in range(self.x_grid):
                    vsk.noiseSeed(np.random.randint(1e8))
                    
                    for point in xy_points:
                        x, y = point
                        
                        # Debug stuff:
                        if self.grid: vsk.circle(x, y, radius=0.02)
                            
                        f = f_swirls[i*self.x_grid + j]
                                    
                        # Draw field:
                        if self.field:
                            dx, dy = f(0, point)
                            v = np.array([dx, dy])
                            v_norm = np.linalg.norm(v)
                            if v_norm > self.norm_max:
                                v = self.norm_max * v / v_norm
                            vsk.line(x, y, x + v[0], y + v[1])
                
                        # Integrate:
                        if self.particles:
                            trace_trajectory(vsk, f, self.N_traj_steps, x, y, self.dt, x_bounds=x_bounds, y_bounds=y_bounds)
                            if self.backward: trace_trajectory(vsk, f, self.N_traj_steps, x, y, -self.dt, x_bounds=x_bounds, y_bounds=y_bounds)

                    if self.border: draw_border(vsk, x_bounds, y_bounds)
                    
                    vsk.translate(self.x_dist, 0)
            vsk.translate(0, self.y_dist)
                
                    

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    PhaseSketch.display()
