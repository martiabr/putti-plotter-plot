Generative art piece inspired by David Rudnick's Schlagenheim art. General idea is to
1) generate connected shape or probability distribution where we want to draw, can do this with shapely probably, define the shape using circles, regular polygons, rectangles, Perlin noise blobs, ellipses, ... and
2) 2) draw randomized shapes on top of eachother with occult plugin to create a chaotic blob of shapes. Randomized shapes can again be circles, regular polygons, rectangles, also more complicated shapes, lines or sticks or antenna like things, also just more complicated concave polygons like lightning bolt type shapes or things with spikes/teeth, sawblades, add partial shading, possible with multiple angles, even full coloring. Maybe define some shapes as more likely far away from border and others more likely closer to border... 
Important aspect is shading, if everything is shaded could add white circles as "holes". Writing an algorithm for shading a general polygon might be challenging. Could even try to color by hand with marker after plotting. Dot shading also interesting to try.

Extra ideas:
- Partial fills e.g. circle a la 6 feet
- Other fills e.g. dot grid, dot grid with noise, short lines (requires collision checks)
- lines
