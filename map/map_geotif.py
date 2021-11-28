from matplotlib import pyplot as plt
import numpy as np
import rasterio as rio
from scipy.ndimage import gaussian_filter

img_1 = rio.open('map/jotunheimen_vest/dtm10/data/dtm10_6801_4_10m_z33.tif').read(1)
img_2 = rio.open('map/jotunheimen_vest/dtm10/data/dtm10_6801_3_10m_z33.tif').read(1)
img = np.vstack((img_1[:-50], img_2))

# plt.figure()
# heatmap = plt.imshow(img, cmap='hot', interpolation='nearest')
# plt.show()

sigma = 1.5 * np.ones(2)
img = gaussian_filter(img, sigma)

x_0 = 0
y_0 = 2426
x_1 = 5044
y_1 = 9494
img = img[y_0:y_1, x_0:x_1]

levels = 60

plt.figure(figsize=(2*2.1,2*2.97))
plt.contour(img, levels=levels, colors='k', linewidths=0.2)
plt.gca().invert_yaxis()
plt.axis('equal')

plt.show()