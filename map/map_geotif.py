from matplotlib import pyplot as plt
import numpy as np
import rasterio as rio
from map_helpers import draw_border, draw_contour_plot, draw_map, show_map_image, trunc_map_lower, smoothen_map, crop_map, interp_map

# img_1 = rio.open('map/jotunheimen_vest/dtm10/data/dtm10_6801_4_10m_z33.tif').read(1)
# img_2 = rio.open('map/jotunheimen_vest/dtm10/data/dtm10_6801_3_10m_z33.tif').read(1)
# img = np.vstack((img_1[:-50], img_2))

# # plt.figure()
# # heatmap = plt.imshow(img, cmap='hot', interpolation='nearest')
# # plt.show()

# sigma = 1.5 * np.ones(2)
# img = gaussian_filter(img, sigma)

# x_0 = 0
# y_0 = 2426
# x_1 = 5044
# y_1 = 9494
# img = img[y_0:y_1, x_0:x_1]

# levels = 60

# plt.figure(figsize=(2*2.1,2*2.97))
# plt.contour(img, levels=levels, colors='k', linewidths=0.2)
# plt.gca().invert_yaxis()
# plt.axis('equal')

# plt.show()










img = rio.open('map/valle/dtm10/data/dtm10_6501_2_10m_z33.tif').read(1)
# show_map_image(img)

img = crop_map(img, [3600, 4000], [300, 860])

skips = 1
img = img[::skips,::skips]

img = interp_map(img, 1)
print('interp done')

img = smoothen_map(img, 0.1)

img = trunc_map_lower(img, 1e-4)
print('trunc done')


show_map_image(img)

draw_contour_plot(img, levels=30)

for x,y in np.ndindex(img.shape):
    if x % 5 == 0 and y % 5 == 0 and img[x, y] < 1:
        plt.scatter(y, x, s=0.01, c='k')

plt.show()