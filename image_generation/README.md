Blender works only on `compute-0-28`. I have tried a few others but not exhaustively. The singularity module for it is in the same `blender` folder as I shared on `/projects/katefgroup/sajaved/blender/`

```bash
srun -w compute-0-28 --time=24:00:00 --gres gpu:1 -c1 --mem=8g --pty $SHELL
IMG="tensorflow-ubuntu-16.04.3-nvidia-375.26_wzy.img"
module load singularity
singularity shell --writable -B /projects:/projects --nv $IMG
```

Here's the command I usually run (replace the first part of the generation command with your blender location):
```bash
cd image_generation
$BLENDER_PATH --background --python render_images.py -- --save_blendfiles 1 --num_images 1000 --max_objects 3 --use_gpu 1 --save_depth_maps 1  --start_idx 0
```

Parser arguments at the top of the `render_script.py` has all the options with their description.

Some of them I've had to change over time are:

1. Object settings:
max_objects, radius, scene_size, all_views

2. Output settings:
start_idx, num_images, output_*_dir, output_scene_file, save_depth_maps, 

3. Rendering settings
use_gpu, width, height, train_flag


The code flow broadly (can be seen in `render_scene_with_tree` function):

1. Set the initial direction of the camera which will become the 0,0
2. Get objects from the tree and put them on the scene
3. Compute the relationship between them
4. Dump binvox and schematic files
5. Dump images from all views
6. Dump depths from all views
7. Compute bboxes from the tree
8. Write the tree and scene json


A few notes:

1. Blender on matrix GPU starts giving errors and stops rendering after writing ~1k samples. This doesn't happen on CPU, so I haven't really looked into it. I simply restart my generation process from the last written sample (using --start_idx 998) once I see it has started printing error messages.

2. When I take the camera to azimuth 90, it is dumped as azimuth 0 to align with Ricson's convention.

3. In `add_objects_from_tree` function the `sample_tree_flexible` function fetches a tree according to which generation happens. This function takes `max_num_objs`, but if needed, you can fix the no of objects in all scenes using the argument `fix_num_objs` instead. Just uncomment and replace with the line below this function call.
