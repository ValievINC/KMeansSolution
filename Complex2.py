import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from skimage.io import imsave, imread
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage import img_as_ubyte
from collections import Counter


def make_an_image(img, num):
    img = np.array(img, dtype=np.float64) / 255
    width, height, depth = tuple(img.shape)
    image_array = np.reshape(img, (width * height, depth))
    image_array_sample = shuffle(image_array, n_samples=1000)
    kmeans = KMeans(n_clusters=num).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    center_colors = kmeans.cluster_centers_
    return labels, center_colors, recreate_image(center_colors, labels, width, height)


def make_a_pie(labels, center_colors):
    counts = Counter(labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_color = [rgb2hex(ordered_colors[i]).upper() for i in counts.keys()]
    plt.figure(figsize=(20, 15))
    plt.pie(counts.values(), labels=hex_color, colors=center_colors)
    plt.savefig('colors.jpg')


def recreate_image(codebook, labels, w, h):
    return codebook[labels].reshape(w, h, -1)


image, colors_num = input().split(' ')
colors_num = int(colors_num)
image = imread(image)
lab, cent_clrs, new_img = make_an_image(image, colors_num)
make_a_pie(lab, cent_clrs)
imsave('newimage.jpg', img_as_ubyte(new_img))
