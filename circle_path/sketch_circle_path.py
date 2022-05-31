import vsketch
import numpy as np


class CirclePathSketch(vsketch.SketchClass):
    # Sketch parameters:
    # radius = vsketch.Param(2.0)

    def draw(self, vsk: vsketch.Vsketch) -> None:
        vsk.size("a4", landscape=False)
        vsk.scale("cm")

        # implement your sketch here
        # vsk.circle(0, 0, self.radius, mode="radius")
        
        f = 0.01
        a = 10.0
        
        for t in range(314):
            x = a * np.sin(f * t)
            y = a * np.cos(f * t)
            vsk.circle(x, y, radius=1.0)

    def finalize(self, vsk: vsketch.Vsketch) -> None:
        vsk.vpype("linemerge linesimplify reloop linesort")


if __name__ == "__main__":
    CirclePathSketch.display()
