# IDEAS:

- Play with attractors, chaotic differential equations in XY plane, also possibly nonlinear diff eqs that are time-varying. Lorenz is the natural place to start.
- Random walks with different constraints/weight functions on position, velocity, acceleration, jerk. Could for instance penalize deviation from some trajectory such as circle or more crazy orbits.
- Grids of objects. Can be as simple as polygons or circles with randomly sampled sizes. Or e.g. create a cactus creator algorithm that generates random cacti (or any other random fun object). Or even just some abstract shapes consisting of points, lines and boxes. Rocket generator would be awesome. Robots. Sail boats.
- Flow fields. Should write general functionality for plotting FFs and then easily change out Perlin noise or nonlinear differential equations etc.
- Trigonometric parametric curves.
- Fun with hexagons. E.g. hexagon perspective pattern with fills?
- Download height map data and generate contour plots for fun geographical locations. Tilezen is one data source. Could try to generate the contour lines custom with Bezier curves and math, or maybe use matplotlib and export to svg. Maybe mpl first to see how it could look and possibly expand. Can retrieve the contour lines from mpl and do things with them if wanted. 
- Experiment with occult plugin to do 2D occlusion. 
- Occult -> 3d isometric drawing with towers, bridges, stairs, ... typical doodle but randomized and parametrized. Possibly procedural algorithm involved?
- Experiment with shapely and occult to do 3D occlusion.
- Flow imager
- Truchet tiles (circles, knots, ...)
- 3D "potential fields" lines (think Currents artwork)
- Experiment with markers/colored pens/multiple layers
- **MASSIV map series. Either just map or also line and dots to show path somehow.**
- DNT map series - add cabins and paths and stuff somehow? Investigate if UT has API of some sorts. Series of different national parks.
- Map library with geotif. Autoscaling to fit a4. Crop function to fit a4. More easily prototype with low resolution then go to real. 
- **Isometric occult tests:** 1. simplex noise to generate cubic terrain, houses on land (randomly placed or somehow find nice places to place buildings algorithmically. 2. city generator - roads and skyscrapers/towers (possibly using wave collapse algorithm (https://marian42.de/article/wfc/) or something else)
- Bauhaus - random shapes, (parallel) lines, multi-color, ... Possibly based on randomly generated quadtree looking grid?
- Single line plot
  - Multi color random walk
- Play with new shapes feature
- Write code for shading shapes. Combined with occult and multiple layers this could create cool effect of random shapes on top of eachother. Would also be cool in black. 
- Path from overlapping circles. Either with or without occlusion

### TODO:
- librarify iso
