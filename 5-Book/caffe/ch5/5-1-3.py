from skimage import io
import sys
import os

folder = sys.argv[1]

l = os.listdir(folder)

item_id = 0
for item in l:
    img = io.imread(folder + item, as_grey=True)
    item_id += 1
print item_id