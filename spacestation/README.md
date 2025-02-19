Look at KSP space stations
Centrifuge?
Can variable width line be used at all?
Remember that vsketch Shapes will be necessary here to create more intricate shapes.
Here we can also add a shapely geometry directly to the shape. So we can just create things in shapely and add as a shape to draw it.

Elements in the space stations:
- Solar panels (single panel directly outwards, double panels directly outwards, or single/double panel on arm, or many panels on either side of an arm)
  - Different connection types
  - Add arm only for connecting solar panels like the ISS?
- Capsules of different sizes and designs
  - Antennas, windows, cargo doors
- Circular inflatable modules?
- Docking ports/bays, 
  - 1. actual structures
  - 2. visual flair on capsule variations
- Bars like in Freedom 
- Centrifuge ring (but hard to draw in 2d, either 4 modules or just a ring. Try to get 3d effect with nonuniform lines, should be also to figure out that math)
- Crew capsules/cabins (capsules with windows)
- Robot arms, antennas, ...
- communication dish tower
- Possibly a spacecraft docked in a bay?


### Algorithm:
- list of structures
- a structure has a width and height, which is added to the bounding_geometry to keep track of occupied space.
      this is used to check if we can place a new structure.
- when placing a structure we also sample a string of points along the edges where new structures may be placed.
- when finding the next structure to be placed, we simply pick a random point. 
- some points may have extra weight to them to get desired behaviour?
- when placing a new structure we update the bounding geometry with a union and the valid next points with a subtract.
- each point must also have a direction associated with it to know how to place the next structure.
- we must also check for collision with outer bounds of the entire space station. This is quite easy to do without having to
      use shapely. Just check if each edge of the bb is within the bounds. 

- the most difficult part is how to structure the data with all the open points, and their associated data (direction and weight).
- e.g. a capsule will only have one open point on the end, but many above and below. Then need to add weight so the point
      on the end will still have a good chance of being picked. 
- Are there any alternative ways of doing it?
      - Only pick the sides and then sample amongst the points. Then weights can be dropped. 
      - Then when the list of points is empty the side will be removed from consideration.
      - The main problem is however how we do the intersection to find points which must be removed and then dropping their 
        data as well. One option is that the xy position is a key in a dict. Works ok but a bit hacky. 
        thinking about this a bit more, if a structure is valid it will only remove points from the "previous" structure, 
        i.e. the one it extends out from. This means we can avoid having a multipoint with all possible points
        instead only the structure itself has a multipoint with its open points which is updated as new structures build out from it.
        How do we pick new points? We maintain a list of all open sides, on format (idx, dir), so we can then pick a random
        side and look it up. As the structure is added and open points are removed we also delete the side if it is empty.
        Afterwards weights could be added if necessary. But weighting each side and each point the same should work ok.

### TODO:
- [x] width, height
- [x] check on bounding geom
- [x] check on outer bb
- [x] add picking between different structure types with different probs
- [x] add end stop structure type
- [x] add weights to encourage going in same direction
- [x] Add extra open point on first capsule
- [x] Remove all the different open points, just choose center. Easy way to force symmetries.
- [x] Add solar panel
- [x] add system for connections between capsules
- [x] Docking bay should not be larger than capsule it is connected to. This should apply to some types. Not solar and capsule, but dock and similar things. The easy solution to this is just to either 1. apply a max to the height sampling, or perhaps better is 2. have height gain that applies to the prev structure height. 
- [x] Connection width matching.  
- [x] for structure types like solar panel and capsule, add subclasses where the variables are overriden. 
      Then it would be possible to first have probs for capsule, solar panel, extra thing etc. 
      And all the little variations of each type can be hidden away inside a second prob density for each type.
      E.g. single panel vs. double panel vs. single/double panel w/wo arm
- [x] Add constraint system. To make it look more like a space station we might want to force symmetries. 
      E.g. if we add a solar panel on one side it should be a high prob that a solar panel will be created on opposite side.
- [x] Quadtree grid
- [ ] Way to only sample a subset of all structure types. Otherwise things will look very chaotic. E.g. there is a 20% chance of having inflatable capsule as an option for a drawing, but a separate prob for actually drawing it when picking the next structure to add. Another example would to be only pick n out of m solar panel types for one drawing, again to make the single drawing more cohesive.
- [x] limit certain parameters (solar panels especially) to be sketch-wide instead of sampled for every module
- [x] Option to not draw "loose" connectors

Modules:
- [x] double panel variation with more beams, probably just as prob in same class
- [x] docking bay
- [x] square capsule with big window/docking bay?
- [x] decoration: just a line sticking out, optional dot
- [x] Variation on square docking bay capsule with lines between outer and inner circle
- [x] Double normal line capsule - just 2x2 normal lines vs. 2xline shaded vs. 2xblack vs. 2xn normal lines
- [x] Antenna: little square end
- [x] Capsule: just some randomly sampled normal lines
- [x] parallel lines capsule - just 1, 2, ... uniformly distributed or "shading" with a sine, or sine + normal lines
- [x] Capsule: multi window, option with parallel lines on top/bottom?
- [x] MultiWindow: dividing lines vertically in offset space
- [x] MultiWindow: rounded windows
- [x] DockingBay: shaded box in addition to the black box
- [x] Decoration: really rounded corners inflatable capsule (actually decoration since no building out from it)
- [x] Decoration: Just a box, or multiple boxes. Centered, or for example one box on either end. Or a tiny box on the box. Small variations like that. Or line out from the box.
- [x] Connector: narrow black connector, height is just gain of the smallest side. Also line shading.
- [x] Capsule: add equal distant normal lines, e.g. a module with two equal distant lines is quite distinctive.
- [x] Connector: flat side + trapezoid side, possibly with flat side shaded or black. 
- [ ] Capsule: normal lines, instead of symmetric on either side, just have two lines next to each other at a random x. Also with line shading or black shading!
- [ ] Decoration: Cupola
- [ ] Connector: Fat ring that is fatter than both start and end, maybe with other black/shaded rings on the sides
- [ ] Randomize antenna position
- [ ] Solar panel: improve single solar panel with arm/connector
- [ ] Capsule: multi small window (long and narrow capsule with dot windows)
- [ ] Capsule: add small boxes to some normal lines, parallel lines capsules, as well as "empty" capsule
- [ ] Antenna: actual little dish on the rod
- [ ] DockSimple: trapezoid
- [ ] Inflatable: normal lines with slight bend, just use ellipse arc
- [ ] Connector: long construction beam type
- [ ] Connector: flat part in middle, so it goes in on both sides and has a long narrow "capsule" in the middle
- [ ] other solar panel arm variations
- [ ] two double panels sticking out from arm
- [ ] Decoration: Robotic arm
- [ ] Decoration: Rocket motor
- [ ] Solar panel: radiators? Would be cool to keep it accurate, but how to draw in 2d?


### Adding weights:
Atm we loop over the structures, add all sides to a list, with idx. However, the weights must be maintained over time.
Or do they? For now the purpose of the weights are only to favour continuing in the same direction. And the direction we know.
So just add to the function that builds the sides. And change to maintain over time if not sufficient later.

### Connections:
Atm there is no consistency from capsule to capsule.
There should be a small prob that this is the case, but in most cases we will need to add a connection.
This is strictly from one capsule to another. For other combinations we add custom interactions.
This means when adding a new capsule and prev is capsule, extend the capsule width by some delta, where the connection will be.
The connection has a min/max angle. The delta width is then determined by the randomly sampled angle + the delta height.
In addition there should be a probability that the same width is used and no connection is added.
Depending on the width of the connection, different styles can be drawn. E.g. if it is long enough we can add windows.

Or we include connection as a separate structure type (subclass of capsule)?
Want to include many different types, also connection with a flat part.
Yes, connection is a separate type, with the special behaviour that it only adds a single open point, in same direction as it has.
To do this right we also need the constraint system. As 1. connection can only be placed on capsule type and 2. only capsule can be placed on connection type (?)

Furthermore, we need the height matching logic to work. The procedure here will be: 1. prev structure side will match height of prev structure, 2.other side will be randomly sampled based on max/min height + sampled gain on prev height. Then the bb size is determined from max of these two. 

One way to do it is to just have a matrix where row=from module, col=to module with probs
Then if from=capsule, to=connector has some prob. If from=connector then to=connector is 0 prob but to=capsule has large prob. And if from is something else, then to=connector is 0.
Should have a default value for every module that the matrix is filled up with
Then augment with extra probs like from=connector and to=connector is 0 etc.
How to do this in a principled way?
You have 1. a list of default probs which are independent of from, and 2. a list of (from, to, prob) which you loop over to overwrite.
However, how important is this really? 
We have capsules, connectors, solar panels and other stuff. That is it really. So you can just give this 4x4 matrix, it is not so difficult...

Ok, this is what we do.
We give two matrices - probs from-to in parallel directions, and probs from-to in normal directions. This way we can for instance have larger prob on solar panels in normal direction from capsules.
only thing as that the rows for solar panel etc. will be redundant and not used... We could have nans instead and then this implies we dont go from this? This simplfies the interface somewhat.

# Width matching:
Atm we have a class method that samples a random width and height given a direction.
Here we can add the prev module to limit the max width. Is there any other information that is useful from prev module? Better is maybe just to input a max height directly.
Then instead of sampling between min, max height we clamp the max to the input max height.

# Per station attributes:
The structure right now is that the constructor is always the same. 
So inputing things in the init is not so easy.
A hack would be to for every iteration in the grid to call update probs and update classes, and it will roll some dice to figure out the parameters and which classes will not be considered.
Then the options string arrays need to be class variables instead of member variables.
Then just a huge job to go through each module type and move out all variables we want to be the same for entire space station.

For the probs we already input the types and the probs. So it should be straightforward to just pick a subset of all modules based on probs for each main moduel type and their respective submodules.
