import sys
import numpy as np
from nbtschematic import SchematicFile

voxel_file = sys.argv[1]
voxel_size = int(sys.argv[2])
sf = SchematicFile.load(voxel_file)
blocks = np.frombuffer(sf.blocks, dtype=sf.blocks.dtype)
blocks = blocks.reshape((voxel_size,voxel_size,voxel_size))
blocks = np.moveaxis(blocks, [0,1,2], [1,0,2])
temp_np_file = voxel_file.split('.schematic')[0] + '.npy'
np.save(temp_np_file, blocks)