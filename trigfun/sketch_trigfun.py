import vsketch
import numpy as np
from tqdm import trange


# - Write math?
#   - Both the params for each instance and write the general equation somewhere?
# - Random rotation?
# - Dont do same parameter configuration twice
# - Why sometimes zero?


def f_sincos(t, a=1, b=1, c=1, d=1, e=1, f=1, g=1, h=1):
    return np.sin(a*t)**e * np.cos(b*t)**f, np.cos(c*t)**g * np.sin(d*t)**h


def f_spiro(t, a, b, c, d, e, f, g, h, exp_a=1.0, exp_b=1.0, exp_c=1.0, exp_d=1.0):
    return a * np.cos(b*t)**exp_a + c * np.cos(d * t)**exp_b, e * np.sin(f*t)**exp_c - g * np.sin(h * t)**exp_d


def scale_path(x_path, y_path, scale=2.0):
    x_range = np.abs(np.max(x_path) - np.min(x_path))
    y_range = np.abs(np.max(y_path) - np.min(y_path))
    x_path = scale * x_path / x_range
    y_path = scale * y_path / y_range
    return x_path, y_path


def center_path(x_path, y_path):
    x_range = np.abs(np.max(x_path) - np.min(x_path))
    y_range = np.abs(np.max(y_path) - np.min(y_path))
    x_path -= np.min(x_path) + 0.5 * x_range
    y_path -= np.min(y_path) + 0.5 * y_range
    return x_path, y_path


class TrigfunSketch(vsketch.SketchClass):
    n_x = vsketch.Param(6)
    n_y = vsketch.Param(9)
    
    N = vsketch.Param(600)
    
    grid_dist_x = vsketch.Param(3.0)
    grid_dist_y = vsketch.Param(3.0)
    
    scale = vsketch.Param(1.00)
    
    f_type = vsketch.Param("sincos", choices=["sincos", "lissajous", "spiro", "spiro_general_1", "spiro_general_2"])
    
    show_math = vsketch.Param(True)
    
    debug = vsketch.Param(True)
    
    random_rot = vsketch.Param(False)
    auto_scale = vsketch.Param(True)
    auto_center = vsketch.Param(True)
    
    a_low = vsketch.Param(1.0)
    a_high = vsketch.Param(6.0)
    b_low = vsketch.Param(1.0)
    b_high = vsketch.Param(6.0)
    c_low = vsketch.Param(1.0)
    c_high = vsketch.Param(6.0)
    d_low = vsketch.Param(1.0)
    d_high = vsketch.Param(6.0)
    e_low = vsketch.Param(1.0)
    e_high = vsketch.Param(6.0)
    f_low = vsketch.Param(1.0)
    f_high = vsketch.Param(6.0)
    g_low = vsketch.Param(1.0)
    g_high = vsketch.Param(6.0)
    h_low = vsketch.Param(1.0)
    h_high = vsketch.Param(6.0)
    exp_a = vsketch.Param(1.0)
    exp_b = vsketch.Param(1.0)
    exp_c = vsketch.Param(1.0)
    exp_d = vsketch.Param(1.0)
    
    t_stop_pi = vsketch.Param(2.0)
    
    rounding_factor = vsketch.Param(4)
    
    rng = np.random.default_rng(123)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale)
        
        self.t = np.linspace(0, self.t_stop_pi*np.pi, self.N)

        for y in trange(self.n_y):
            with vsk.pushMatrix():
                for x in range(self.n_x):
                    
                    # Parameter random choice:
                    a, b, c, d, e, f, g, h = 1, 1, 1, 1, 1, 1, 1, 1
                    if self.f_type == "sincos":
                        straight_line = True
                        while straight_line:
                            a = self.rng.integers(self.a_low, self.a_high + 1)
                            b = self.rng.integers(self.b_low, self.b_high + 1)
                            c = self.rng.integers(self.c_low, self.c_high + 1)
                            d = self.rng.integers(self.d_low, self.d_high + 1)
                            straight_line = (a == d) and (b == c)
                    elif self.f_type == "spiro":
                        a = np.round(self.rng.uniform(self.a_low, self.a_high) * self.rounding_factor) / self.rounding_factor
                        b = np.round(self.rng.uniform(self.b_low, self.b_high) * self.rounding_factor) / self.rounding_factor
                    elif self.f_type in ("lissajous", "spiro_general_2"):
                        a = self.rng.integers(self.a_low, self.a_high + 1)
                        b = self.rng.integers(self.b_low, self.b_high + 1)
                        c = self.rng.integers(self.c_low, self.c_high + 1)
                        d = self.rng.integers(self.d_low, self.d_high + 1)
                        e = self.rng.integers(self.e_low, self.e_high + 1)
                        f = self.rng.integers(self.f_low, self.f_high + 1)
                        g = self.rng.integers(self.g_low, self.g_high + 1)
                        h = self.rng.integers(self.h_low, self.h_high + 1)
                    elif self.f_type == "spiro_general_1":
                        a = np.round(self.rng.uniform(self.a_low, self.a_high) * self.rounding_factor) / self.rounding_factor
                        b = np.round(self.rng.uniform(self.b_low, self.b_high) * self.rounding_factor) / self.rounding_factor
                        c = np.round(self.rng.uniform(self.c_low, self.c_high) * self.rounding_factor) / self.rounding_factor
                        d = np.round(self.rng.uniform(self.d_low, self.d_high) * self.rounding_factor) / self.rounding_factor
                        e = np.round(self.rng.uniform(self.e_low, self.e_high) * self.rounding_factor) / self.rounding_factor
                        f = np.round(self.rng.uniform(self.f_low, self.f_high) * self.rounding_factor) / self.rounding_factor
                        g = np.round(self.rng.uniform(self.g_low, self.g_high) * self.rounding_factor) / self.rounding_factor
                        h = np.round(self.rng.uniform(self.h_low, self.h_high) * self.rounding_factor) / self.rounding_factor

                    # Generate data:
                    if self.f_type == "sincos":
                        x_path, y_path = f_sincos(self.t, a, b, c, d, self.exp_a, self.exp_b, self.exp_c, self.exp_d)
                    elif self.f_type == "spiro":
                        x_path, y_path = f_spiro(self.t, (1 - a), 1.0, a * b, (1 - a) / a, (1 - a), 1.0, a * b, (1 - a) / a)
                    elif self.f_type == "spiro_general_1":
                        x_path, y_path = f_spiro(self.t, a, b, c, d, e, f, g, h, self.exp_a, self.exp_b, self.exp_c, self.exp_d)
                    
                    if self.auto_scale:
                        x_path, y_path = scale_path(x_path, y_path)  
                        
                    if self.auto_center:                  
                        x_path, y_path = center_path(x_path, y_path)  
                    
                    if self.debug:
                        self.show_debug_text(vsk, a, b, c, d, e, f, g, h)
                    
                    sketch = vsketch.Vsketch()
                    sketch.detail(1e-2)
                    
                    with vsk.pushMatrix():
                        if self.random_rot:
                            sketch.rotate(self.rng.uniform(0, 2*np.pi))
                        for i in range(1, self.N):
                            sketch.line(x_path[i-1], y_path[i-1], x_path[i], y_path[i])
                    vsk.sketch(sketch)
                    
                    vsk.translate(self.grid_dist_x, 0)    
            vsk.translate(0, -self.grid_dist_y)
            
        if self.show_math:
            text = ""
            if self.f_type == "sincos":
                if e == 1 and f == 1 and g == 1 and h == 1:
                    text = "x(t) = sin(at) cos(bt),  y(t) = sin(ct) cos(dt)"
            elif self.f_type == "sincos":
                text = "x(t) = sin(at) cos(bt),  y(t) = sin(ct) cos(dt)"
            elif self.f_type == "spiro_general_1":
                text = "x(t) = a sin(bt) + c cos(dt),  y(t) = e sin(ft) + g cos(ht)"
            vsk.text(text, y=self.n_y * self.grid_dist_y + 1.8, x=16.0, size=0.3, align="right", font="rowmans")
    
    def show_debug_text(self, vsk, a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0):
        vsk.stroke(2)
        vsk.circle(0, 0, radius=5e-2)
        text = []
        if self.f_type in ("sincos", "sincos2"):
            text.append(f"a={a}, b={b}, c={c}, d={d}")
        elif self.f_type == "spiro":
            text.append(f"a={a:.2f}, b={b:.2f}")
        elif self.f_type in ("spiro_general_1", "spiro_general_2"):
            text.append(f"a={a:.2f}, b={b:.2f}, c={c:.2f}, d={d:.2f}")
            text.append(f"e={e:.2f}, f={f:.2f}, g={g:.2f}, h={h:.2f}")
        for i, t in enumerate(text):
            vsk.text(t, y=1.2 + 0.2 * i, size=0.14, align="center", font="rowmans")
        vsk.stroke(1)
         
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    TrigfunSketch.display()
