import os
import binvox_rw

fname = 'CLEVR_new_000003'
filepath = os.path.join('obj_files', fname + '.binvox')

with open(filepath, 'rb') as f:
	model = binvox_rw.read_as_3d_array(f)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(model.data)
plt.show()
# 3d_filepath = os.path.join('obj_files', fname + '.png')
# plt.savefig(3d_filepath)
