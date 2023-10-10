from HOG_algorythm import get_hog as my_hog
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt
# import seaborn as sns

im = cv2.imread("Pics/4.png", cv2.IMREAD_GRAYSCALE)
fd, hog_image = hog(im, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True)

my_fd = my_hog(im)

# sns.set_style("darkgrid")

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 20))
axs[0].plot(fd, color='green')
axs[0].set_title("skimage HOG")
axs[1].plot(my_fd, color='r')
axs[1].set_title("My awesome HOG")


# plt.figure(figsize=(10, 20))
# plt.subplot(2, 1, 1)
# plt.plot(fd, color='g')

# plt.subplot(2, 1, 2)
# plt.plot(my_fd, color='r')

plt.show()