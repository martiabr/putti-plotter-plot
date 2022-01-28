from inspect import trace
import vsketch
import numpy as np
from scipy.integrate import solve_ivp

# Initial idea:
# Flow fields, phase plot, attractors, perlin noise, nonlinear time-varying dynamics
# 
# Want to be able to either visualize grid of derivatives of draw phase plane trajectories (integrate)
# The underlying dynamics should be seperated from the flow field viz,
# such that it should be easy to change out perlin noise based flow field for nonlinear chaos system or combination.
# Scipy should be fine for integrating? Or just RK4.

# TODO:
# - express system in polar coordinates
# - draw grid and border
# - randomized particle positions
# - heuristic for picking good particle positions
# - integrate backwards as well?
# - better control of scaling and center 
# - menu of different fs
# - menu of different styles of grid
# - menu of different styles of paths
# - customize noise
# - Add custom parameters a la scipy
# - Fix function mess
# - Draw flow field function

def draw_border(vsk, x_bounds, y_bounds):
    vsk.line(x_bounds[0], y_bounds[0], x_bounds[1], y_bounds[0])
    vsk.line(x_bounds[0], y_bounds[1], x_bounds[1], y_bounds[1])
    vsk.line(x_bounds[0], y_bounds[0], x_bounds[0], y_bounds[1])
    vsk.line(x_bounds[1], y_bounds[0], x_bounds[1], y_bounds[1])

def integrate_euler(x, f, dt):
    return x + dt * f(x)


def integrate_RK4(x, f, dt):
    k1 = f(x)
    k2 = f(x + dt * k1 / 2.0)
    k3 = f(x + dt * k2 / 2.0)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        
def trace_trajectory(vsk, f, traj_length, x_0, y_0, dt, x_bounds=None, y_bounds=None):
    z = np.zeros((traj_length+1,2))
    z[0] = np.array([x_0, y_0])
    for i in range(traj_length):
        z[i+1] = integrate_RK4(z[i], f, dt)
        
        if x_bounds is not None and (z[i+1,0] < x_bounds[0] or z[i+1,0] > x_bounds[1]):
            break
        if y_bounds is not None and (z[i+1,1] < y_bounds[0] or z[i+1,1] > y_bounds[1]):
            break
        
        vsk.line(z[i,0], z[i,1], z[i+1,0], z[i+1,1])

class PhaseSketch(vsketch.SketchClass):
    
    border = vsketch.Param(True)
    grid = vsketch.Param(False)
    x_min = vsketch.Param(4.0)
    x_max = vsketch.Param(4.0)
    y_min = vsketch.Param(4.0)
    y_max = vsketch.Param(4.0)
    
    f_n = vsketch.Param(0.5)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        


        # def f_scipy(t, z):
        #     angle = 2 * np.pi * vsk.noise(x_bounds[0] + f_n*x, y_bounds[0] + f_n*y)
        #     dx = 10 * np.cos(angle)
        #     dy = 10 * np.sin(angle)
        #     return [dx, dy]
            
        def f_perlin(z):
            x, y = z
            angle = 2 * np.pi * vsk.noise(x_bounds[0] + self.f_n*x, y_bounds[0] + self.f_n*y)
            dx = np.cos(angle)
            dy = np.sin(angle)
            return np.array([dx, dy])
        
        def f_pendulum(z):
            dx = z[1]
            dy = -k * np.sin(z[0]) - d * z[1]
            return np.array([dx, dy])
        
        def f_experiment(z):
            # dx = z[1]
            # dy = - z[0] + 0.5*z[1]
            # dy = 0.05*z[1] + np.cos(z[0] + z[1])**2
            # dx = -2.0 * z[0] - 2 * z[1] * np.abs(1 - z[1])
            # dy = 1.0 * z[0] * np.abs(1 - z[0]) + 1.0 * z[1]
            x, y = z
            dx = y
            dy = - 0.1 * (x - 1)**3 - 0.4 * np.abs(np.sin(y))
            return np.array([dx, dy])
        
            
        N = 20
        k = 1
        d = 1
        
        scale = 0.1
        x_bounds = np.array([-self.x_min, self.x_max])
        y_bounds = np.array([-self.y_min, self.y_max])
        x_range = np.linspace(x_bounds[0],x_bounds[1],N)
        y_range = np.linspace(y_bounds[0],y_bounds[1],N)
        for x in x_range:
            for y in y_range:
                # Debug stuff:
                if self.grid:
                    vsk.circle(x, y, radius=0.02)
                
                # Derivative function:
                # dx = y
                # dy = -k * np.sin(x) - d * y
                
                # The basic way is to normalize vector such that only angle is given by perlin noise...
                angle = 2 * np.pi * vsk.noise(x_bounds[0] + self.f_n*x, y_bounds[0] + self.f_n*y)
                dx = np.cos(angle)
                dy = np.sin(angle)
            
                
                # Draw field:
                # Just direction, max norm, ...
                # v = scale * np.array([dx, dy])
                # v_norm = np.linalg.norm(v)
                # v_norm_max = 0.2
                # if v_norm > v_norm_max:
                #     v = v_norm_max * v / v_norm
                # vsk.line(x, y, x + v[0], y + v[1])
        
                # Integrate:
                # t = np.linspace(0, 0.4, 51)
                # sol = solve_ivp(f_scipy, (0, 0.4), [x, y], t_eval=t)
                # for i in range(50):
                    # vsk.line(sol.y[0,i], sol.y[1,i], sol.y[0,i+1], sol.y[1,i+1])
                # Ok maybe it is not that simple. Should probably do the integration ourself. And take more care with designing field?
                # odeint is probably also overkill when we want to do many lines. It will take forever to solve. RK4 is probably faster.
                # Or just pick faster solver
                
                dt = 4e-2
                traj_length = 4
                trace_trajectory(vsk, f_perlin, traj_length, x, y, dt, x_bounds=x_bounds, y_bounds=y_bounds)
                trace_trajectory(vsk, f_perlin, traj_length, x, y, -dt, x_bounds=x_bounds, y_bounds=y_bounds)

        if self.border: draw_border(vsk, x_bounds, y_bounds)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    PhaseSketch.display()
