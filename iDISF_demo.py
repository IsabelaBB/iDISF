import sys
sys.path.append("./python3/");

from idisf import iDISF_scribbles
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

#img = np.array(Image.open("./208001.jpg"), dtype=np.int32);
img = np.array(Image.open("../datasets/GeoStar_dataset/images/189080.jpg"), dtype=np.int32);

n0 = 50
iterations = 1
f=2
c1=1.0
c2=1.0
num_obj = 1
segm_method = 1
all_borders = 1

#path_markers = './scribbles_208001.txt'
path_markers = '../iDISF/markers/user_seeds_geostar/189080-anno.txt'
file = open(path_markers, 'r')
lines = file.readlines()

coords = []
size_scribbles = []
size_scribbles.append(int(lines[1]))

index_sizes = 0
acum=0
for i in range(2,len(lines)):
    if(acum == size_scribbles[index_sizes]):
        acum=0
        index_sizes+=1
        size_scribbles.append(int(lines[i]))
    else:
        coords.append([int(n) for n in lines[i].split(';')])
        acum+=1

#coords = [[50,30], [100,200], [150,80]]
#size_scribbles = [1,2]

coords = np.array(coords)
size_scribbles = np.array(size_scribbles)

label_img,border_img = iDISF_scribbles(img, n0, iterations, coords, size_scribbles, num_obj, f, c1, c2, segm_method, all_borders);

fig = plt.figure();
fig.add_subplot(1,2,1);
labelplot = plt.imshow(label_img, cmap = 'gray', vmin = 0, vmax = np.max(label_img));
fig.add_subplot(1,2,2);
borderplot = plt.imshow(border_img ,cmap='gray', vmin = 0, vmax = np.max(border_img));
plt.show();


