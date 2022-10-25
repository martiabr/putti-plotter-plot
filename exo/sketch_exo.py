from pyparsing import col
import vsketch
import pandas as pd

class ExoSketch(vsketch.SketchClass):
    # Sketch parameters:
    # radius = vsketch.Param(2.0)
    
    def get_col_names(self, file_path):
        col_names = pd.read_csv(file_path, names=["Abbrv", "Column name"], delimiter=":", skiprows=5, nrows=77)
        col_names["Abbrv"] = col_names["Abbrv"].str.replace("# COLUMN ", "")
        col_names["Abbrv"] = col_names["Abbrv"].str.rstrip()
        col_names["Column name"] = col_names["Column name"].str.strip()
        return col_names
    
    def process_data(self, file_path):
        col_names = self.get_col_names(file_path)
        # print(col_names)
        
        df = pd.read_csv(file_path, header=83, nrows=10000)
        
        # TODO:
        # - Remove doubles (keep most recent?)
        #   - For all unique names, find all with the name, pick most recent update?
        #   - That is the easiest heuristic
        #   - Better would probably be to combine the rows somehow. So we get the least amount of NaNs?
        # - Group items by star somehow
        # - Filter away items that dont have data in x
        # - Filter away items with too few planets
        # - Filter away not confirmed?
        # - Create plots of the data! E.g. histograms
        
        min_num_planets = 2
        col_name = col_names[col_names["Column name"] == "Number of Planets"]["Abbrv"]
        selection = df[col_name] >= min_num_planets
        df = df.loc[selection[col_name].to_numpy()]
        
        # print((df[df["soltype"] != "Published Confirmed"]))
        
        
        
        print(df)
        

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")

        # implement your sketch here
        # vsk.circle(0, 0, self.radius, mode="radius")
        
        self.process_data("data/PS_2022.10.24_10.45.43.csv")

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    ExoSketch.display()
