We want a solution that is general. So setting the tileset and the ruleset should be independent of the actual wfc code.
We have a boolean 3D array of size N_x x N_y x N_tiles that keeps track of every possibility.

---

- https://www.boristhebrave.com/2020/04/13/wave-function-collapse-explained/
- https://www.gridbugs.org/wave-function-collapse/
- https://www.procjam.com/tutorials/wfc/
- https://robertheaton.com/2018/12/17/wavefunction-collapse-algorithm/
- https://terbium.io/2018/11/wave-function-collapse/
- https://www.jackbrewer.co.uk/plot/truchet-tiles
- https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.378.5320&rep=rep1&type=pdf#page=69
- https://www.skillshare.com/projects/Truchet-Tiles-in-Python-Mode-for-Processing/390887

---

### Rules:

The ruleset consists of the following type of rules:
1. Tile A cannot be above Tile B
2. Tile A cannot be next to Tile B (four of rule 1.)
3. Tile A must be below Tile B
4. Tile A must be next to Tile B

These rules override the starting assumption that all tiles can go next to eachother.

How do we store the information of a rule? Tile A, Tile B, black/white list, direction
How do we store the complete ruleset? Just a list/tuple of rules?

How to check rules in nice way?
The way we check is by grouping rules by tiles, so all rules that needs to be satisified for this grid cell to be a Tile A, a Tile B and so on. This indicates that the rules should rather be stored in 2D list?

---

### Tileset:

Tileset consists of the actual drawings and weights.

Shannon Entropy tells us the entropy for every gridcell. Always pick lowest entropy.
E = - sum(p_i log(p_i))
When the probability is represented by a weight we get:
E = log(sum(weights)) - sum(w_i log(w_i)) / sum(weights)
If we normalize the weights we can use the above expression!

---

### Collapse:

We have a queue of cells to update after picking cell with lowest entropy.
Add neighbors to queue if their probability space updated. FIFO queue must be better.
Iterate until queue empty.

---

How to do this?
- Calculate entropy for all grid cells (we could do some caching here)
  - function for calculating entropy at x,y
  - function for calculating entropy for all
  - later improve this by on new iteration by only update entropy on updated cells after all propagation is done.
- Pick cell with lowest entropy and collapse its state
  - function for picking cell
    - how to we avoid the already collapsed cells? simply hacky solution is to set entropy to nan or inf
    - alternative is to first filter above 0 and then take min.
    - do this first as it is simple but need to check how slow this is later...
  - function for collapsing it
- Propagation: check neighbours if their possiblity space changes, if so, add their neighbours again to check and so on until the queue is empty
  - The check is done by looping through all rules for all still valid tile choices for a given grid cell. If any rules are broken, short circuit and set possibility to False and continue testing for all the remaining valid tile choices. 
  - A flag is used to indicate that the cell's possibility space updated and if flag is set neighbours are added.
  - Make sure to ignore already collapsed cells
- Repeat

- Pop queue
- Get valid tiles for cell
- For each tile
  - Get all rules for tile
  - For each rule
    - Dont check rule if dir sends us outside bounds
    - Check rule
      - must be false:
        - only check that other cell in DIR of cell is not collapsed and not collapsed to tile
      - must be true:
        - check that tile is still in possible tiles for other cell

---

# TODO:
- [x] Make loop better
- [x] Debug draw grid
- [x] Debug draw tile index in corner
- [x] Optional debug print
- [x] Draw tiles
- [x] Function for generating tileset nicely
- [x] Fix sampling of tiles using prob
- [x] Randomize queue 
- [x] Check speed
- [ ] Numba? Need to move some functions outside class to speed up
- [x] Extend tileset
- [ ] Improve tileset
  - [ ] Variations (dots)
  - [ ] Thickness
  - [ ] Optional overlap
  - [ ] Variable width between sections
  - [ ] Think about more variations
- [x] Tune probs, the rotated ones need lower prob
- [x] Edge constraints
- [x] Progress bar
- [ ] **Speed up by having variations of same tile. The only unique thing here is the valid directions. This will give much more maintainable speeds as adding variations will not make things slower**
- [ ] Change to depth first search
- [ ] Depth to search per iteration? How to do this without breaking any rule logic?
- [ ] Iterate through and draw between each
- [ ] Start from center?
- [ ] Deal with fails - fallback, restart or backtrack?
- [ ] Add easy random init of partially filled tileset based on other rules
- [ ] Duplicating tiles on several layers, tiles may belong to zero, one, or multiple "categories" which correspond to each layer.
  - [ ] Constraint on number of "connected" groups from each category/layer? 

---

It gets stuck in propagate, adds and pops forever.
Possibly because same cell is added multiple times in queue?
Would it be correct to check and not add cell to queue if already there? Or delete the one we already have?

Problem: we are trying to remove possibilties that are already removed... And by doing this we say things are updated when they are not.

The code seems to work without the reversed stuff so maybe it works... Must be further investigated

---

Line profiler on propagate:
- 17% get_valid_directions()
- 31% dir_to_cell()
- 14+21% simple if checks
So propagate is not really issue?
No all time of iterate is used in propagate.
Draw uses nothing.
Just need to reduce checks somehow...

---

### Ideas:
- Experiment with adding noise to tiles for a more chaotic look.
- Experiment with bezier curves and possible 2 or 3 lines on each tile edge for a more choatic "shoelace" look 
- "Infinite circuit" drawing?
- Generative metro map
- Inka?
- Remove small loops

### Extending to diagonals:
Atm we 
- collapse cell
- add 4 neighbours to Q
- pop n from Q
- Get valid tile possibilities and valid directions from n
- For every possible tile, for every rule, check if any rules are broken, and if so set tile possibility to false and break 
- If updated, update entropy and add valid neighbours to Q

All of this should be possible to extend to all 8 neighbours? But it does increase complexity
