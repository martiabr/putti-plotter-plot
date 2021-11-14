import shapefile
import os
import numpy as np
import shapely.geometry as geo
import matplotlib.pyplot as plt

path = "map/fyn/"
dirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

all_lines = []
for dir in dirs:
    with shapefile.Reader(path + dir + "/" + dir) as shp:
        # lines = [geo.shape(shape_i.shape.__geo_interface__) for shape_i in shp]
        
        # print(shp)
        
        lines = []
        for shape_i in shp:
            shape = geo.shape(shape_i.shape.__geo_interface__)
            if shape.type == geo.MultiLineString:
        #         print(line.type)
                for line in shape:
                    lines.append(line)
            else:
                lines.append(shape)
        all_lines.append(lines)
        
# Plot:
plt.figure()
for lines in all_lines:
    for line in lines:
        # print(*line.xy)
        # print(line.type)
        try:
            plt.plot(*line.xy, color='k', linewidth=0.5)
        except NotImplementedError:
            print('failed')
plt.show()