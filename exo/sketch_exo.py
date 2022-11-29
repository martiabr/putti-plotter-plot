import vsketch
import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
# from plotter_util import draw_shaded_circle


def draw_shaded_circle(vsk, x_0, y_0, radius, fill_distance, angle):
    vsk.circle(x_0, y_0, radius=radius)
    N = np.max((0, int(np.round(2 * (radius / fill_distance - 1)))))
    fill_distance = 2 * radius / (N + 1)
    with vsk.pushMatrix():
        vsk.translate(x_0, y_0)
        vsk.rotate(angle)
        for d in np.linspace(-radius + fill_distance, radius - fill_distance, N, endpoint=True):
            dy = radius * np.sin(np.arccos(d / radius))
            vsk.line(d, -dy, d, dy) 
                   

class ExoSketch(vsketch.SketchClass):
    # Sketch parameters:
    
    scale = vsketch.Param(7.0)
    n_x = vsketch.Param(3, min_value=1)
    n_y = vsketch.Param(5, min_value=1)
    grid_dist_x = vsketch.Param(0.75)
    grid_dist_y = vsketch.Param(0.75)
    
    min_num_planets = vsketch.Param(2, min_value=1)
    
    random_angles = vsketch.Param(True)
    planet_line_padding = vsketch.Param(0.00)
    star_radius_gain = vsketch.Param(0.010)
    planet_radius_gain = vsketch.Param(0.003)
    
    draw_multi_stars = vsketch.Param(True)
    min_multi_star_orbit_radius_gain = vsketch.Param(1.2)
    max_multi_star_orbit_radius_gain = vsketch.Param(1.6)
    
    draw_inclination = vsketch.Param(False)
    inclination_scale = vsketch.Param(1.0)
    
    use_aop = vsketch.Param(True)
    
    # Scaling:
    system_scaling = vsketch.Param("Log", choices=["None", "Log", "Normalize"])
    system_normalized_size = vsketch.Param(0.22, min_value=0.0)
    log_base = vsketch.Param(3.0, min_value=0.0)
    
    # Fill:
    filled_planets = vsketch.Param(False)
    planets_fill_angle = vsketch.Param(0.75)
    planets_fill_thickness = vsketch.Param(0.005)
    filled_stars = vsketch.Param(False)
    stars_fill_angle = vsketch.Param(0.75)
    stars_fill_thickness = vsketch.Param(0.005)

    # Text:
    planet_font_size = vsketch.Param(0.035)
    system_font_size = vsketch.Param(0.060)
    font_padding_star = vsketch.Param(0.020)
    font_padding_planet = vsketch.Param(-0.010)
    system_font_type = vsketch.Param("rowmans", choices=["rowmand", "rowmans", "rowmant", "futuram", "timesi",
                                                  "timesib", "timesr", "timesrb"])
    planet_font_type = vsketch.Param("timesi", choices=["rowmand", "rowmans", "rowmant", "futuram", "timesi",
                                                  "timesib", "timesr", "timesrb"])
    
    def get_col_names(self, file_path):
        col_names = pd.read_csv(file_path, names=["Abbrv", "Column name"], delimiter=":", skiprows=5, nrows=75)
        col_names["Abbrv"] = col_names["Abbrv"].str.replace("# COLUMN ", "")
        col_names["Abbrv"] = col_names["Abbrv"].str.strip()
        col_names["Column name"] = col_names["Column name"].str.strip()
        return col_names
    
    def process_data(self, file_path, cols_filter=["pl_orbeccen", "pl_orbsmax", "st_rad", "pl_rade"], min_num_planets=3):
        col_names = self.get_col_names(file_path)
        
        df = pd.read_csv(file_path, header=109)
        print(f"Read {len(df)} planets from {df.hostname.nunique()} systems.")
        
        # TODO:
        # - Create plots of the data! E.g. histograms
        # - Add dict manually for doing GJ->Gliese, Cnc-> Cancri, KOI->...->Kepler to get nicer names to print
        # - Get data on stars as well? Especially for binary sytems...
        # - Get composition data - draw plots?
        
        # def col_name_to_abbrv(col_name):
        #     return col_names[col_names["Column name"] == col_name]["Abbrv"]
        
        # selection = df["sy_pnum"] >= min_num_planets
        # df = df[selection.to_numpy()]
        # print(f"Reduced dataset to {len(df)} planets from {df.hostname.nunique()} systems with {min_num_planets} or more planets.")        
        
        # print((df[df["soltype"] != "Published Confirmed"]))
        
        df = df[df[cols_filter].notnull().all(1)]
        print(f"Reduced dataset to {len(df)} planets from {df.hostname.nunique()} systems with non-nan values for columns {cols_filter}.")        
        
        counts = df["hostname"].value_counts()
        df = df[df["hostname"].isin(counts[counts > min_num_planets].index)]
        print(f"Reduced dataset to {len(df)} planets from {df.hostname.nunique()} systems with {min_num_planets} or more planets after filtering.")        
        
        df = df.reset_index(drop=True)
        
        # print(df["hostname"].value_counts().to_string())
        
        df["pl_orblper"] = df["pl_orblper"].fillna(0)
        
        # Replace some shortened words to full versions:
        name_translations = {"Cnc": "Cancri", "GJ": "Gliese", "tau Cet": "Tau Ceti"}
        # TODO: V ... Tau -> Tauri
        for k, v in name_translations.items():
            df["pl_name"] = df["pl_name"].str.replace(k, v)
            df["hostname"] = df["hostname"].str.replace(k, v)
            
        return df
    
    def draw_system(self, vsk, x_star, y_star, system):
        with vsk.pushMatrix():            
            num_stars = system.iloc[0]["sy_snum"]
            
            star_radius = self.star_radius_gain * system.iloc[0]["st_rad"]
            # star_spectral_type = system.iloc[0]["st_spectype"]
            star_name = system.iloc[0]["hostname"]
            
            # TODO: move some of these computations to process_data
            
            system["pl_name_short"] = system["pl_name"].str.replace(star_name, "")
            system["pl_name_short"].str.strip()
            
            system["pl_orbsmin"] = system.apply(lambda row: row["pl_orbsmax"] * np.sqrt(1.0 - row["pl_orbeccen"]**2), axis=1)
            system["pl_orbfoci"] = system.apply(lambda row: row["pl_orbsmax"] * row["pl_orbeccen"], axis=1)
            # system["y_dist"] = system.apply(lambda row: self.planet_radius_gain * row["pl_rade"] + row["pl_orbsmax"] * np.sin(np.deg2rad(row["pl_orblper"])), axis=1)
            system["y_dist"] = system.apply(lambda row: self.planet_radius_gain * row["pl_rade"] + row["pl_orbsmin"], axis=1)
            
            largest_orbit_planet = system.loc[system["y_dist"].idxmax()]
            c_max = largest_orbit_planet["pl_orbfoci"]
            y_dist_max = largest_orbit_planet["y_dist"]
            
            # Compute scale factor:
            scale_factor = 1.0
            if self.system_scaling == "Normalize":
                scale_factor = self.system_normalized_size / y_dist_max
            elif self.system_scaling == "Log":
                scale_factor = np.log(self.system_normalized_size / y_dist_max + 1.0) / np.log(self.log_base)
                        
            system_sketch = vsketch.Vsketch()
            system_sketch.detail("1e-3")
            
            with system_sketch.pushMatrix():
                # Draw star(s):
                if self.draw_multi_stars:
                    pos_stars = np.zeros((num_stars, 2))
                    if num_stars > 1:
                        star_orbit_radius_gain = np.random.uniform(self.min_multi_star_orbit_radius_gain, self.max_multi_star_orbit_radius_gain)
                        star_orbit_radius = star_orbit_radius_gain * scale_factor * star_radius
                        start_angle = np.random.uniform(0, 2 * np.pi)
                        delta_angle = 2 * np.pi / num_stars
                        for i_star in range(num_stars):
                            angle_i = start_angle + delta_angle * i_star
                            pos_stars[i_star] = star_orbit_radius * np.array([np.cos(angle_i), np.sin(angle_i)])
                else:
                    pos_stars = np.zeros((1, 2))
                
                for star_pos in pos_stars:    
                    if self.filled_stars:
                        draw_shaded_circle(system_sketch, star_pos[0], star_pos[1], radius=scale_factor * star_radius, fill_distance=self.stars_fill_thickness,
                                        angle=self.stars_fill_angle)
                    else:
                        system_sketch.circle(star_pos[0], star_pos[1], radius=scale_factor * star_radius)
                
                # Draw planets and orbits:
                for index, planet in system.iterrows():
                    planet_name = planet["pl_name_short"]
                    
                    e = planet["pl_orbeccen"]
                    planet_radius = self.planet_radius_gain * scale_factor * planet["pl_rade"]
                    a = scale_factor * planet["pl_orbsmax"]
                    b = scale_factor * planet["pl_orbsmin"]
                    c = scale_factor * planet["pl_orbfoci"]
                    aop = np.deg2rad(planet["pl_orblper"])
                    # TODO: deal with aop
                    
                    if self.random_angles:
                        theta = np.random.uniform(0, 2 * np.pi)
                    else:
                        theta = 0.0
                    
                    with system_sketch.pushMatrix():
                        if self.use_aop:
                            system_sketch.rotate(aop)
                        
                        angle_offset = (self.planet_line_padding + 2 * planet_radius) / (a + b)
                        system_sketch.arc(-c, 0, 2 * a, 2 * b, theta + angle_offset, 2 * np.pi + theta - angle_offset)
                        
                        x, y = a * np.cos(theta) - c, - b * np.sin(theta)
                        if self.filled_planets:
                            draw_shaded_circle(system_sketch, x, y, planet_radius, fill_distance=self.planets_fill_thickness,
                                            angle=self.planets_fill_angle)
                        else:
                            system_sketch.circle(x, y, radius=planet_radius)
                        
                        # Planet label:
                        system_sketch.text(f"{planet_name}", x=(x + planet_radius + self.font_padding_planet), y=y,
                            font=self.planet_font_type, size=self.planet_font_size, align="left", mode="label")

                    # Inclination drawing:
                    if self.draw_inclination:
                        with system_sketch.pushMatrix():
                            inclination = np.deg2rad(planet["pl_orbincl"])
                            system_sketch.scale(self.inclination_scale)
                            system_sketch.rotate(inclination)
                            planet_line_pos = np.random.uniform(-2 * a, 2 * a) - c
                            system_sketch.circle(planet_line_pos, 0, planet_radius)
                            system_sketch.line(-c - 2 * a, 0, -c + 2 * a, 0)

            # System label:
            system_sketch.text(f"{star_name.upper()}", x=0, y=(scale_factor * y_dist_max + 0.5 * self.system_font_size + self.font_padding_star),
                               font=self.system_font_type, size=self.system_font_size, align="center", mode="transform")
            
            vsk.translate(x_star, y_star)   
            vsk.translate(0, -(scale_factor * y_dist_max + 0.5 * self.system_font_size + self.font_padding_star))
            vsk.sketch(system_sketch)
        
    def get_habitable_planets(self):
        url = "https://en.wikipedia.org/wiki/List_of_potentially_habitable_exoplanets"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': "wikitable"})
        df = pd.read_html(str(table))
        return df

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")
        vsk.scale(self.scale)

        df = self.process_data("data/PSCompPars_2022.10.28_09.15.11.csv", min_num_planets=self.min_num_planets)
        # print(df)
        systems = df.groupby("hostname")
        
        # df_hab = self.get_habitable_planets()
        # TODO: df_hab -> indices in df
        
        N_systems = self.n_x * self.n_y
        
        random_range = np.arange(systems.ngroups)
        np.random.shuffle(random_range)
        systems = df[systems.ngroup().isin(random_range[:N_systems])].groupby("hostname")
        for i, (star_name, system) in enumerate(systems):
            x = i % self.n_x
            y = i // self.n_x
            self.draw_system(vsk, x * self.grid_dist_x, -y * self.grid_dist_y, system)
        
    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    ExoSketch.display()
