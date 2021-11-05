from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
import numpy as np
import laspy
import os
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Scramble?
# Text
# Fix water height

skips = 100
chunk_size = 1000000
# path = 'map/bymarka/data/'
path = 'map/bymarka_2/data/'
# path = 'map/trondheim/data/'

file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# all in one go:
process_data = False
if process_data:
    coords = np.zeros((0,3))
    for file_name in file_names:
        las = laspy.read(path + file_name)
        coords_all = np.vstack((las.x, las.y, las.z)).transpose()
        np.random.shuffle(coords_all)
        coords = np.vstack((coords, coords_all[::skips]))
        print(file_name + ' processed.')
    # np.save('map/coords_bymarka.npy', coords)
    np.save('map/coords_bymarka_2.npy', coords)
    # np.save('map/coords_trondheim.npy', coords)
else:
    # coords = np.load('map/coords_bymarka.npy')
    coords = np.load('map/coords_bymarka_2.npy')
    # coords = np.load('map/coords_trondheim.npy')
    print('Data loaded with shape', coords.shape)

# Fix water levels:
z_min = 1.0
for i, point in enumerate(coords):
    if point[2] < z_min:
        coords[i,2] = z_min

# chunks:
# coords = np.zeros((0,3))
# for file_name in file_names:
#     with laspy.open(path + file_name) as f:
#         for points in f.chunk_iterator(chunk_size):
#             chunk_coords = np.vstack((points.x, points.y, points.z)).transpose()
#             np.random.shuffle(chunk_coords)
#             coords = np.vstack((coords, chunk_coords[::skips]))
#             print(coords.shape)
# print(coords)

# Generate a regular grid to interpolate the data:
N = 2000
xmin = np.min(coords[:,0])
xmax = np.max(coords[:,0])
ymin = np.min(coords[:,1])
ymax = np.max(coords[:,1])

xmin = 561400
xmax = 568300
ymin = 7031400
ymax = 7036400

xi = np.linspace(xmin, xmax, N)
yi = np.linspace(ymin, ymax, N)

# Interpolate using delaunay triangularization :
zi = griddata((coords[:,0], coords[:,1]),coords[:,2], (xi[None,:], yi[:,None]), method='nearest')
print('Interpolation completed.')

# Filter using Gaussian filter:
# sigma = 6 * np.ones(2)
sigma = 4.5 * np.ones(2)
z_smooth = gaussian_filter(zi, sigma)
# TODO: test other smoothers
print('Smoothing completed.')

# Plot contour:
levels = 50
levels_final = 120
# levels_final = 100

# plt.figure()
# plt.scatter(coords[:,0], coords[:,1], c=coords[:,2], s=0.02)

# plt.figure()
# cs = plt.tricontourf(coords[:,0], coords[:,1], coords[:,2], levels=levels)

# plt.figure()
# plt.tricontour(coords[:,0], coords[:,1], coords[:,2], levels=levels, colors='k', linewidths=0.5)

# plt.figure()
# plt.contour(xi, yi, zi, levels=levels, colors='k', linewidths=0.5)

# plt.figure()
# cs = plt.contourf(xi, yi, z_smooth, levels=levels_final)

# X, Y = np.meshgrid(xi, yi)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, z_smooth, cmap='jet')

heatmap = plt.imshow(z_smooth, cmap='hot', interpolation='nearest')
plt.colorbar(heatmap)

plt.figure()
cs = plt.contour(xi, yi, z_smooth, levels=levels_final, colors='k', linewidths=0.5)

# Note: possible with array of linewidths

# for tick in cs.xaxis.get_major_ticks():
#     tick.tick1line.set_visible(False)
#     tick.tick2line.set_visible(False)
#     tick.label1.set_visible(False)
#     tick.label2.set_visible(False)
# for tick in cs.yaxis.get_major_ticks():
#     tick.tick1line.set_visible(False)
#     tick.tick2line.set_visible(False)
#     tick.label1.set_visible(False)
#     tick.label2.set_visible(False)
plt.gca().axis('off')

# plt.savefig('map/output/bymarka.svg')
plt.savefig('map/output/bymarka_2.svg')
# plt.savefig('map/output/trondheim.svg')

plt.show()
