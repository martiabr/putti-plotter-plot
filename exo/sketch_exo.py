from pyparsing import col
import vsketch
import pandas as pd

class ExoSketch(vsketch.SketchClass):
    # Sketch parameters:
    # radius = vsketch.Param(2.0)
    
    def get_col_names(self, file_path):
        col_names = pd.read_csv(file_path, names=["Abbrv", "Column name"], delimiter=":", skiprows=5, nrows=77)
        col_names["Abbrv"] = col_names["Abbrv"].str.replace("# COLUMN ", "")
        col_names["Abbrv"] = col_names["Abbrv"].str.strip()
        col_names["Column name"] = col_names["Column name"].str.strip()
        return col_names
    
    def process_data(self, file_path, cols_filter=["pl_orbincl", "pl_orbper"], min_num_planets=3):
        col_names = self.get_col_names(file_path)
        # print(col_names)
        
        df = pd.read_csv(file_path, header=83, nrows=10000)
        df["rowupdate"] = pd.to_datetime(df["rowupdate"])
        
        # TODO:
        # - Remove doubles (keep most recent?)
        #   - For all unique names, find all with the name, pick most recent update?
        #   - That is the easiest heuristic
        #   - Better would probably be to combine the rows somehow. So we get the least amount of NaNs?
        #   - Easy to do groupby + mean
        # - Group items by star somehow
        # - Filter away items that dont have data in x
        # - Filter away items with too few planets
        # - Filter away not confirmed?
        # - Create plots of the data! E.g. histograms
        
        # def col_name_to_abbrv(col_name):
        #     return col_names[col_names["Column name"] == col_name]["Abbrv"]
        
        selection = df["sy_pnum"] >= min_num_planets
        df = df[selection.to_numpy()]
        
        # print((df[df["soltype"] != "Published Confirmed"]))
        
        df = df[df[cols_filter].notnull().all(1)]
        
        print(df)
        
        # Remove duplicates:
        # df = df.groupby("pl_name").mean().reset_index()  # merge using mean
        df = df.loc[df.groupby("pl_name").rowupdate.idxmax()]  # pick newest
        
        print(df)
        
        print(df.groupby("hostname"))
        

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
