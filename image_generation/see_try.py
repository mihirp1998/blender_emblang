import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from nbtschematic import SchematicFile
from mpl_toolkits.mplot3d import Axes3D
res = 64
vpath8 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000001.schematic'
vpath9 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000060.schematic'

vpath5 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000029.schematic'
vpath6 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000034.schematic'
vpath7 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000039.schematic'

vpath1 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000161.schematic'
vpath2 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000181.schematic'
vpath3 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000050.schematic'
vpath4 = '../output/CLEVR_' + str(res) + '_OBJ_FULL/voxels/train/CLEVR_new_000035.schematic'
vpaths = [vpath8, vpath9, vpath5, vpath6, vpath7, vpath1, vpath2, vpath3, vpath4]
for vpath in vpaths:
	sf = SchematicFile.load(vpath)
	blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
	data = np.frombuffer(sf.data, dtype=sf.data.dtype)
	blocks = blocks.reshape((res,res,res))
	vals = np.unique(blocks)
	colors = np.empty(blocks.shape, dtype=object)
	colorname = ['red','blue','green','black','yellow','cyan','magenta']
	for i,c in zip(vals, colorname):
	    colors[blocks == i] = c
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.voxels(blocks, facecolors=colors)
	plt.show()
	plt.close()