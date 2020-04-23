
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import _init_paths
import math, sys, random, argparse, json, os, tempfile, pickle

from datetime import datetime as dt
from collections import Counter
from treeutils import sample_tree, extract_objects, refine_tree_info, remove_function_obj, sample_tree_flexible, add_parent
from modules import Combine, Layout, Describe
from lib.tree import Tree
import pdb
import subprocess
import os
import numpy as np
import time
from mathutils import Matrix
from math import radians
import binvox_rw


"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""
# Made changes to add_objects_from_tree(), render_scene_with_tree()
# Ready for 3d

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import utils

    except ImportError as e:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene_full.blend',
                    help="Base blender file on which all scenes are based; includes " +
                         "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
                    help="JSON file defining objects, materials, sizes, and colors. " +
                         "The \"colors\" field maps from CLEVR color names to RGB values; " +
                         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
                         "rescale object models; the \"materials\" and \"shapes\" fields map " +
                         "from CLEVR material and shape names to .blend files in the " +
                         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
                    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
                    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
                    help="Optional path to a JSON file mapping shape names to a list of " +
                         "allowed color names for that shape. This allows rendering images " +
                         "for CLEVR-CoGenT.")

# Settings for objects
parser.add_argument('--min_objects', default=1, type=int,
                    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=3, type=int,
                    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.2, type=float,
                    help="The minimum allowed distance between object centers")
parser.add_argument('--min_obj_2d_size', default=10, type=float,
                    help="The minimum allowed 2d bounding box size of generated objects")
parser.add_argument('--radius', default=13, type=float,
                    help="The distance of the camera from the origin from where the images are rendered")
parser.add_argument('--scene_size', default=8, type=float,
                    help="The distance of the camera from the origin from where the images are rendered")
parser.add_argument('--all_views', default=1, type=float,
                    help="Render all 36 views or only the 4 needed for testing")
parser.add_argument('--filter_out_of_view', default=0, type=int,
                    help="Reject scenes with out-of-view objects")
parser.add_argument('--allow_floating_objects', default=0, type=int,
                    help="Boolean flag for whether to allow floating objects")
parser.add_argument('--include_inside_config', default=0, type=float,
                    help="Include 'x inside y' scenes ")
parser.add_argument('--percent_inside_samples', default=0.1, type=float,
                    help="Percentage of scenes which will have 'inside' layout"),
parser.add_argument('--back_front_only_flag', default=0, type=int,
                    help="Flag for rendering samples with only configurations 'back' and 'front'")
parser.add_argument('--margin', default=0.0, type=float,
                    help="Along all cardinal directions (left, right, front, back), all " +
                         "objects will be at least this distance apart. This makes resolving " +
                         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=20, type=int,
                    help="All objects will have at least this many visible pixels in the " +
                         "final rendered images; this ensures that no objects are fully " +
                         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
                    help="The number of times to try placing an object before giving up and " +
                         "re-placing all objects in the scene.")
parser.add_argument('--render_from_given_objects', default=0, type=int,
                    help="Flag for rendering samples using given object descriptions. Uses the dictionary" + 
                    "specified in the argument 'given_objects_json_path'")
parser.add_argument('--given_objects_json_path', default='given_objects.json',
                    help="Path for the object descriptions to be used for rendering if " +
                    " the flag 'render_from_given_objects' is on")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
                    help="The index at which to start for numbering rendered images. Setting " +
                         "this to non-zero values allows you to distribute rendering across " +
                         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
                    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
                    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
                    help="Name of the split for which we are rendering. This will be added to " +
                         "the names of rendered images, and will also be stored in the JSON " +
                         "scene structure for each image.")
parser.add_argument('--dataset_name', default='CLEVR_DATASET_DEFAULT',
                    help="Name of the main folder")
parser.add_argument('--output_image_dir', default='../output/{}/images/',
                    help="The directory where output images will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/{}/scenes/',
                    help="The directory where output JSON scene structures will be stored. " +
                         "It will be created if it does not exist.")
parser.add_argument('--output_tree_dir', default='../output/{}/trees/',
                    help="The directory where output trees will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--output_depth_dir', default='../output/{}/depth/',
                    help="The directory where output trees will be stored. It will be " +
                         "created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/{}/CLEVR_scenes.json',
                    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='../output/{}/voxels/',
                    help="The directory where blender scene files will be stored, if the " +
                         "user requested that these files be saved using the " +
                         "--save_blendfiles flag; in this case it will be created if it does " +
                         "not already exist.")
parser.add_argument('--save_depth_maps', type=int, default=0,
                    help="The flag for whether to save a depth map")
parser.add_argument('--save_blendfiles', type=int, default=1,
                    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
                         "each generated image to be stored in the directory specified by " +
                         "the --output_blend_dir flag. These files are not saved by default " +
                         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
                    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
                    default="Creative Commons Attribution (CC-BY 4.0)",
                    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
                    help="String to store in the \"date\" field of the generated JSON file; " +
                         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
                    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
                         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                         "to work.")
parser.add_argument('--width', default=64, type=int,
                    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=64, type=int,
                    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
                    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
                    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
                    help="The number of samples to use when rendering. Larger values will " +
                         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
                    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
                    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
                    help="The tile size to use for rendering. This should not affect the " +
                         "quality of the rendered image but may affect the speed; CPU-based " +
                         "rendering may achieve better performance using smaller tile sizes " +
                         "while larger tile sizes may be optimal for GPU-based rendering.")
parser.add_argument('--train_flag', default=1, type=int,
                    help="generate training or test, set to 0 for testing")
parser.add_argument('--zero_shot', default=0, type=int,
                    help="Whether to use zero-shot setting when generate the data")
parser.add_argument('--add_layout_prob', default=0.5, type=float,
                    help="probability of adding an extra layout layer")


def main(args):
    num_digits = 6
    prefix = '%s_%s_' % (args.filename_prefix, args.split)
    img_template = '%s%%0%dd.png' % (prefix, num_digits)
    scene_template = '%s%%0%dd.json' % (prefix, num_digits)
    obj_template = '%s%%0%dd.obj' % (prefix, num_digits)
    tree_template = '%s%%0%dd.tree' % (prefix, num_digits)
    depth_template = '%s%%0%dd.png' % (prefix, num_digits)

    if args.train_flag != 0:
        args.train_flag = True
    else:
        args.train_flag = False

    if args.zero_shot != 0:
        args.zero_shot = True
    else:
        args.zero_shot = False

    args.output_image_dir = args.output_image_dir.format(args.dataset_name)
    args.output_scene_dir = args.output_scene_dir.format(args.dataset_name)
    args.output_tree_dir = args.output_tree_dir.format(args.dataset_name)
    args.output_depth_dir = args.output_depth_dir.format(args.dataset_name)
    args.output_blend_dir = args.output_blend_dir.format(args.dataset_name)
    args.output_scene_file = args.output_scene_file.format(args.dataset_name)

    if args.train_flag:
        split_output_image_dir = os.path.join(args.output_image_dir, 'train/')
        split_output_tree_dir = os.path.join(args.output_tree_dir, 'train/')
        split_output_scene_dir = os.path.join(args.output_scene_dir, 'train/')
        split_output_blend_dir = os.path.join(args.output_blend_dir, 'train/')
        split_output_depth_dir = os.path.join(args.output_depth_dir, 'train/')
    else:
        split_output_image_dir = os.path.join(args.output_image_dir, 'test/')
        split_output_tree_dir = os.path.join(args.output_tree_dir, 'test/')
        split_output_scene_dir = os.path.join(args.output_scene_dir, 'test/')
        split_output_blend_dir = os.path.join(args.output_blend_dir, 'test/')
        split_output_depth_dir = os.path.join(args.output_depth_dir, 'test/')

    img_template = os.path.join(split_output_image_dir, img_template)
    scene_template = os.path.join(split_output_scene_dir, scene_template)
    obj_template = os.path.join(split_output_blend_dir, obj_template)
    tree_template = os.path.join(split_output_tree_dir, tree_template)
    depth_template = os.path.join(split_output_depth_dir, depth_template)

    if not os.path.isdir(args.output_image_dir):
        os.makedirs(args.output_image_dir)
    if not os.path.isdir(split_output_image_dir):
        os.makedirs(split_output_image_dir)
    if not os.path.isdir(args.output_scene_dir):
        os.makedirs(args.output_scene_dir)
    if not os.path.isdir(split_output_scene_dir):
        os.makedirs(split_output_scene_dir)
    if not os.path.isdir(args.output_tree_dir):
        os.makedirs(args.output_tree_dir)
    if not os.path.isdir(split_output_tree_dir):
        os.makedirs(split_output_tree_dir)

    if args.save_depth_maps == 1 and not os.path.isdir(split_output_depth_dir):
        os.makedirs(split_output_depth_dir)

    if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
        os.makedirs(split_output_blend_dir)

    all_images = list(sorted(os.listdir(split_output_image_dir)))
    if len(all_images) > 0:
        max_idx = int(all_images[-1][10:])
        start_idx = max_idx + 1
    else:
        start_idx = args.start_idx

    all_scene_paths = []
    for i in range(args.num_images):
        img_path = img_template % (i + start_idx)
        scene_path = scene_template % (i + start_idx)
        tree_path = tree_template % (i + start_idx)
        all_scene_paths.append(scene_path)
        obj_path = None
        depth_path = None
        if args.save_blendfiles == 1:
            obj_path = obj_template % (i + start_idx)
        if args.save_depth_maps == 1:
            depth_path = depth_template % (i + start_idx)

        while True:
            try:
                render_scene_with_tree(args,
                                       tree_max_level=3,
                                       output_index=(i + start_idx),
                                       output_split=args.split,
                                       output_image=img_path,
                                       output_scene=scene_path,
                                       output_blendfile=obj_path,
                                       output_tree=tree_path,
                                       depth_path=depth_path
                                       )
                break
            except Exception as e:
                print(e)
                import traceback
                print(traceback.print_tb(e.__traceback__))
                import shutil
                import glob
                if os.path.exists(img_path.replace('.png','')):
                    shutil.rmtree(img_path.replace('.png',''))
                if os.path.isfile(scene_path):
                    os.remove(scene_path)
                voxel_files_del = glob.glob(obj_path.replace('.obj','*'))
                for file in voxel_files_del:
                    os.remove(file)
                if os.path.isfile(scene_path):
                    os.remove(scene_path)
                if depth_path is not None and os.path.exists(depth_path.replace('.png','')):
                    shutil.rmtree(depth_path.replace('.png',''))
                # print(img_path, scene_path, obj_path, tree_path, depth_path)
                # import sys
                # sys.exit()


    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, 'r') as f:
            all_scenes.append(json.load(f))
    output = {
        'info': {
            'date': args.date,
            'version': args.version,
            'split': args.split,
            'license': args.license,
        },
        'scenes': all_scenes
    }
    with open(args.output_scene_file, 'w') as f:
        json.dump(output, f)


'''

    Tree-based generation

'''

def get_calibration_matrix_K_from_blender(camd):
    from mathutils import Matrix
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio 
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # print(resolution_x_in_px)
    # print(resolution_y_in_px)
    # print(scene.render.pixel_aspect_x)
    # print(scene.render.pixel_aspect_y)
    # print(f_in_mm)
    # print(s_u, s_v)
    # print(pixel_aspect_ratio)
    # print(sensor_width_in_mm)
    # print(sensor_height_in_mm)
    # import sys
    # sys.exit()

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K



# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
         ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K*RT, K, RT


def get_calib():
    from mathutils import Matrix
    from math import tan

    scn = bpy.data.scenes['Scene']
    cam = bpy.data.cameras['Camera']
    w = scn.render.resolution_x*scn.render.resolution_percentage/100.
    h = scn.render.resolution_y*scn.render.resolution_percentage/100.

    C = Matrix().to_3x3()
    C[0][0] = -w/2 / tan(cam.angle/2)
    ratio = w/h
    C[1][1] = -h/2. / tan(cam.angle/2) * ratio
    C[0][2] = w / 2.
    C[1][2] = h / 2.
    C[2][2] = 1.
    C.transpose()
    print(cam.angle)
    return C


def render_scene_with_tree(args,
                           tree_max_level=3,
                           output_index=0,
                           output_split='none',
                           output_image='render.png',
                           output_scene='render_json',
                           output_blendfile=None,
                           output_tree='tree.tree',
                           depth_path=None,
                           ):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    # render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    # Setup CompositeNodeMapRange for depth maps if flag is on
    if args.save_depth_maps:
        bpy.context.scene.use_nodes = True
        blender_tree = bpy.context.scene.node_tree
        blender_tree.nodes.new('CompositorNodeMapRange')
        # Range of depth (Hacky values set accordind to the location of the cameras for this project)
        blender_tree.nodes["Map Range"].inputs["From Min"].default_value = 0
        blender_tree.nodes["Map Range"].inputs["From Max"].default_value = 100      

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        'split': output_split,
        'image_index': output_index,
        'image_filename': os.path.basename(output_image),
        'objects': [],
        'directions': {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=args.scene_size)
    plane = bpy.context.object

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects['Camera']

    # Assign 90,30 as the base scene wrt which trees are generated
    # 90, 30 corresponds to 0,30 in Ricson's code
    # camera.location = obj_centered_camera_pos(args.radius, 90.0, 30.0)

    plane_normal = plane.data.vertices[0].normal
    print('The plane normal is: ', plane_normal)
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()


    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    if args.render_from_given_objects:
        # Read the specified objects json and render the given objects
        # with open(args.given_objects_json_path, 'rb') as f:
        #     given_objects = pickle.load(f)
        # add_objects_from_given_trees(args, given_objects)

        with open(args.given_objects_json_path, 'r') as f:
            given_objects = json.load(f)
        add_objects_from_given_trees(args, given_objects)

        # Render the scene for all thetas and phis and dump the scene data structure
        # Ricson's code needs an offset of 90 for the thetas/phis to align.
        offset = 90
        if args.all_views:
            THETAS = list(range(0+offset, 360+offset, 30))
            PHIS = list(range(20, 80, 20))
            PHIS.insert(0, 12)
        else:
            THETAS = list(range(0+offset, 360+offset, 90))
            PHIS = [40]

        image_name = os.path.basename(output_image).split('.png')[0]
        output_image = os.path.join(os.path.dirname(output_image), image_name)

        # Render original view
        render_args.filepath = os.path.join(output_image, image_name + '_orig.png')
        while True:
            try:
                bpy.ops.render.render(write_still=True)
                break
            except Exception as e:
                print(e)

        # Render all other views
        for theta in THETAS:
            for phi in PHIS:
                start = time.time()
                camera.location = obj_centered_camera_pos(args.radius, theta, phi)
                render_args.filepath = os.path.join(output_image, image_name + '_' + str(theta - offset) + '_' + str(phi) + '.png')
                while True:
                    try:
                        bpy.ops.render.render(write_still=True)
                        break
                    except Exception as e:
                        print(e)
        import sys
        sys.exit()
    else:
        # Now make some random objects
        objects, blender_objects, phrase_tree = add_objects_from_tree(scene_struct, args, camera, tree_max_level)

    # Store scene struct
    scene_struct['objects'] = objects
    scene_struct['relationships'] = compute_all_relationships(scene_struct)

    # *****************************************************************************

    '''
    # import bmesh

    # bm = bmesh.new()
    # bmesh.ops.create_monkey(bm)
    # mesh = bpy.data.meshes.new('Monkey')
    # bm.to_mesh(mesh)
    # bm.free()
    # obj = bpy.data.objects.new('Object', mesh)
    # obj.location = (1,1,1)
    # bpy.context.scene.objects.link(obj)
    # bpy.context.scene.update()

    # Debug zone
    thetas = [90]
    phis = [20]

    for theta, phi in zip(thetas, phis):
        filename = str(theta) + '_' + str(phi)

        camera.location = obj_centered_camera_pos(args.radius, theta, phi)

        bpy.ops.wm.save_as_mainfile(filepath='output_blendfile_{}.blend'.format(filename))
        render_args.filepath = 'rendered_image_{}.png'.format(filename)
        bpy.ops.render.render(write_still=True)


        bpy.context.scene.use_nodes = True
        blender_tree = bpy.context.scene.node_tree
        blender_tree.nodes.new('CompositorNodeMapRange')
        blender_tree.nodes["Map Range"].inputs["From Min"].default_value = 0
        blender_tree.nodes["Map Range"].inputs["From Max"].default_value = 100     
        blender_tree.links.new(blender_tree.nodes["Render Layers"].outputs["Depth"], blender_tree.nodes["Map Range"].inputs["Value"])
        blender_tree.links.new(blender_tree.nodes["Map Range"].outputs["Value"], blender_tree.nodes["Composite"].inputs["Image"])
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.data.scenes['Scene'].render.filepath = 'rendered_depth_{}.png'.format(filename)
        bpy.ops.render.render(write_still=True)


        bpy.ops.object.select_all(action='DESELECT')
        for obj in bpy.data.objects:
            if (obj.type == 'MESH' and obj.name != 'Ground'):
                obj.select = True

        bpy.ops.export_scene.obj(
                filepath='output_{}.obj'.format(filename),
                use_selection=True,
                use_materials=True
                )

        # cmd = subprocess.Popen(["binvox", "-d", "64", "-e", "-bb" " -10" " -10" " -10" " 10" " 10" " 10", " output.obj"], stdout=None, close_fds=True)
        command = 'binvox -d 64 -e -bb {} {} {} {} {} {} -t binvox -e -mb output_{}.obj'.format(-args.scene_size, -args.scene_size, -args.scene_size, args.scene_size, args.scene_size, args.scene_size, filename)
        os.system(command)

        command = 'binvox -d 64 -e -bb {} {} {} {} {} {} -t schematic -e -mb output_{}.obj'.format(-args.scene_size, -args.scene_size, -args.scene_size, args.scene_size, args.scene_size, args.scene_size, filename)
        os.system(command)

        os.remove('output_{}.obj'.format(filename))
        os.remove('output_{}.mtl'.format(filename))

        voxel_file = 'output_{}.schematic'.format(filename)
        cmd = subprocess.Popen(["python", "write_voxels.py", voxel_file, str(args.width)], stdout=subprocess.PIPE, close_fds=True)
        output = cmd.communicate(timeout=None)[0]

        # Read the voxels
        temp_np_file = voxel_file.split('.schematic')[0] + '.npy'
        blocks = np.load(temp_np_file)
        block_scene = np.zeros_like(blocks)
        ids = np.unique(blocks)
        for i in ids:
            if i != 0:
                object_idx = np.where(blocks == i)
                x_top = object_idx[0].min()
                y_top = object_idx[1].min()
                z_top = object_idx[2].min()

                x_bottom = object_idx[0].max()
                y_bottom = object_idx[1].max()
                z_bottom = object_idx[2].max()

                block_scene[x_top:x_bottom, y_top:y_bottom, z_top:z_bottom] = 1.0

        translate = [0.0, 0.0, 0.0]
        model = binvox_rw.Voxels(np.transpose(block_scene, [2, 1, 0]) > 0.5, block_scene.shape, translate, 1.0, 'xyz')
        with open('bbox_scene_{}.binvox'.format(filename), 'wb') as f:
            model.write(f)

    import sys
    sys.exit()
    '''

    # *****************************************************************************


    # Keep commented
    # if output_blendfile is not None:
    #     bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

    # Write a temp .obj file for binvox
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if (obj.type == 'MESH' and obj.name != 'Ground'):
            obj.select = True

    bpy.ops.export_scene.obj(
            filepath=output_blendfile,
            use_selection=True,
            use_materials=True
            )


    # Write .schematic file with voxels using binvox and delete .obj files
    if output_blendfile is not None:
        # cmd = subprocess.Popen(["./binvox", "-d", "64", "-t", "schematic", "-e", "-mb", output_blendfile], stdout=subprocess.PIPE, close_fds=True)
        # cmd = subprocess.Popen(["binvox", "-d", "64", "-t", "schematic", "-e", "-mb -bb -10 -10 -10 10 10 10", output_blendfile], stdout=None, close_fds=True)
        # output = cmd.communicate(timeout=None)[0]
        command = './binvox -d 64 -e -bb {} {} {} {} {} {} -t schematic -e -mb {}'.format(-args.scene_size, -args.scene_size, -args.scene_size, args.scene_size, args.scene_size, args.scene_size, output_blendfile)
        os.system(command)
        
        command = './binvox -d 64 -e -bb {} {} {} {} {} {} -t binvox -e -mb {}'.format(-args.scene_size, -args.scene_size, -args.scene_size, args.scene_size, args.scene_size, args.scene_size, output_blendfile)
        os.system(command)

    if os.path.exists(output_blendfile):
        os.remove(output_blendfile)
        os.remove(output_blendfile.split('.obj')[0] + '.mtl')

    voxel_file = output_blendfile.split('.obj')[0] + '.schematic'
    command = 'python write_voxels.py {} {}'.format(voxel_file, str(args.width))
    os.system(command)
    # cmd = subprocess.Popen(["python", "write_voxels.py", voxel_file, str(args.width)], stdout=subprocess.PIPE, close_fds=True)
    # output = cmd.communicate(timeout=None)[0]

    # Read the voxels
    temp_np_file = voxel_file.split('.schematic')[0] + '.npy'
    blocks = np.load(temp_np_file)

    if os.path.exists(temp_np_file):
        os.remove(temp_np_file)

    # Render the scene for all thetas and phis and dump the scene data structure
    # Ricson's code needs an offset of 90 for the thetas/phis to align.
    offset = 90
    if args.all_views:
        THETAS = list(range(0+offset, 360+offset, 30))
        PHIS = list(range(20, 80, 20))
        PHIS.insert(0, 12)
    else:
        THETAS = list(range(0+offset, 360+offset, 90))
        PHIS = [40]

    image_name = os.path.basename(output_image).split('.png')[0]
    output_image = os.path.join(os.path.dirname(output_image), image_name)

    # Remove the dummy objects if there's just one real object before rendering
    bpy.ops.object.select_all(action='SELECT')
    objs = bpy.data.objects
    if len(objects) == 1:
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and obj.name != 'Ground' and obj.scale[0] <= 0.25:
                objs.remove(obj, do_unlink=True)

    render_args.filepath = os.path.join(output_image, image_name + '_orig.png')
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    for theta in THETAS:
        for phi in PHIS:
            start = time.time()
            camera.location = obj_centered_camera_pos(args.radius, theta, phi)
            render_args.filepath = os.path.join(output_image, image_name + '_' + str(theta - offset) + '_' + str(phi) + '.png')
            while True:
                try:
                    bpy.ops.render.render(write_still=True)
                    break
                except Exception as e:
                    print(e)
            view_key = str(theta - offset) + '_' + str(phi)
            scene_struct = get_2d_bboxes(args, camera, scene_struct, view_key)
            print('*'*30)
            print('images')
            print(time.time() - start)
            print('*'*30)

    # Render depth maps if flag is on
    if args.save_depth_maps:
        depth_path = os.path.join(os.path.dirname(depth_path), image_name)
        blender_tree.links.new(blender_tree.nodes["Render Layers"].outputs["Depth"], blender_tree.nodes["Map Range"].inputs["Value"])
        blender_tree.links.new(blender_tree.nodes["Map Range"].outputs["Value"], blender_tree.nodes["Composite"].inputs["Image"])
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        for theta in THETAS:
            for phi in PHIS:
                start = time.time()
                camera.location = obj_centered_camera_pos(args.radius, theta, phi)
                bpy.data.scenes['Scene'].render.filepath = os.path.join(depth_path, image_name + '_' + str(theta - offset) + '_' + str(phi))
                while True:
                    try:
                        bpy.ops.render.render(write_still=True)
                        break
                    except Exception as e:
                        print(e)
                print('*'*30)
                print('depth')
                print(time.time() - start)
                print('*'*30)

    # Refine the tree, remove function_objs
    phrase_tree = refine_tree_info(phrase_tree, blocks)
    phrase_tree = remove_function_obj(phrase_tree)
    # tree = add_parent(tree)

    with open(output_tree, 'wb') as f:
        pickle.dump(phrase_tree, f, protocol=2)

    with open(output_scene, 'w') as f:
        json.dump(scene_struct, f, indent=2)


def get_2d_bboxes(args, camera, scene_struct, view_key):
    """
    Get 2d bboxes for the current camera view
    """
    bpy.ops.mesh.primitive_plane_add(radius=args.scene_size)
    plane = bpy.context.object

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    plane_normal = plane.data.vertices[0].normal

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    direction_dict = {}
    direction_dict['directions'] = {}
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Save all six axis-aligned directions in the scene struct
    direction_dict['directions']['behind'] = tuple(plane_behind)
    direction_dict['directions']['front'] = tuple(-plane_behind)
    direction_dict['directions']['left'] = tuple(plane_left)
    direction_dict['directions']['right'] = tuple(-plane_left)
    direction_dict['directions']['above'] = tuple(plane_up)
    direction_dict['directions']['below'] = tuple(-plane_up)
        

    with open(args.properties_json, 'r') as f:
        properties = json.load(f)

    size_mapping = properties['sizes']

    for i, obj in enumerate(scene_struct['objects']):
        obj_loc = obj['3d_coords']
        obj_name_out = obj['shape']
        size_name = obj['size']
        r = size_mapping[size_name]
        pixel_coords_lefttop, pixel_coords_rightbottom = get_bbox(args, camera, direction_dict, Vector(obj_loc), obj_name_out, r)
        if args.filter_out_of_view:
            assert args.width == args.height
            flag_1 = 0 <= pixel_coords_lefttop[0] < args.width
            flag_2 = 0 <= pixel_coords_lefttop[1] < args.width
            flag_3 = 0 <= pixel_coords_rightbottom[1] < args.width
            flag_4 = 0 <= pixel_coords_rightbottom[1] < args.width
            if not (flag_1 and flag_2 and flag_3 and flag_4):
                raise Exception('Object out of view')

        scene_struct['objects'][i]['bbox_2d'][view_key] = {}
        scene_struct['objects'][i]['bbox_2d'][view_key]['pixel_coords_lefttop'] = pixel_coords_lefttop
        scene_struct['objects'][i]['bbox_2d'][view_key]['pixel_coords_rightbottom'] = pixel_coords_rightbottom
    return scene_struct


def add_objects_from_given_trees(args, objects):
    """
    Add given objects at the given positions
    """
    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = properties['materials']
        object_mapping = properties['shapes']
        size_mapping = properties['sizes']

    for specified_obj in objects[:1]:
        shape = specified_obj['shape']
        size = specified_obj['size']
        color = specified_obj['color']
        material = specified_obj['texture']
        location = specified_obj['location']

        obj_name = object_mapping[shape]
        scale = size_mapping[size]

        # Add the given object at specified location
        obj_name = utils.add_given_object(args, obj_name, scale, location)

        # Add material and color to the given object
        mat_name = material_mapping[material]
        rgba = color_name_to_rgba[color]
        utils.add_material(mat_name, Color=rgba)

def add_objects_from_tree(scene_struct, args, camera, tree_max_level):
    """
    Add random objects to the current blender scene
    """
    # tree = sample_tree(tree_max_level, add_layout_prob=args.add_layout_prob, zero_shot=args.zero_shot, train=args.train_flag)
    # tree = sample_tree_flexible(max_layout_level=2, add_layout_prob=0.6, zero_shot=False, train=True, arguments={'fix_num_objs':2})


    tree = sample_tree_flexible(args.percent_inside_samples, args.include_inside_config, max_layout_level=2, add_layout_prob=0.8, obj_count=0, zero_shot=False, train=True, arguments={'max_num_objs':args.max_objects, 'min_num_objs':args.min_objects}, back_front_only_flag=args.back_front_only_flag)
    # tree = sample_tree_flexible(args.percent_inside_samples, args.include_inside_config, max_layout_level=2, add_layout_prob=0.6, obj_count=0, zero_shot=False, train=True, arguments={'fix_num_objs':args.max_objects}, back_front_only_flag=args.back_front_only_flag)
    specified_objects = extract_objects(tree)

    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = properties['materials']
        object_mapping = properties['shapes']
        size_mapping = properties['sizes']
        print('size mapping:', size_mapping)
        print('object mapping', object_mapping)

    shape_color_combos = None
    if args.shape_color_combos_json is not None:
        with open(args.shape_color_combos_json, 'r') as f:
            shape_color_combos = list(json.load(f).items())

    positions = []
    objects = []
    blender_objects = []

    # Check if the current scene contains inside configuration
    put_obj_inside = False
    stored_location = None
    min_dist = args.min_dist
    if tree.word == 'inside':
        put_obj_inside = True
        min_dist = 0.0

    for obj_counter, specified_obj in enumerate(specified_objects):
        # Choose a random size
        size_name = specified_obj.attributes['size'].attr_val
        # print('\n'*10)
        # print(size_name)
        # print('\n'*10)

        # with open("test_sampled.txt", "a") as myfile:
        #     myfile.write(size_name + '\n')

        r = size_mapping[size_name]

        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            if stored_location:
                break

            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                print(obj_counter)
                return add_objects_from_tree(scene_struct, args, camera, tree_max_level)

            x = specified_obj.position[0] * scene_struct['directions']['right'][0] + specified_obj.position[1] * \
                scene_struct['directions'][
                    'front'][0]
            y = specified_obj.position[0] * scene_struct['directions']['right'][1] + specified_obj.position[1] * \
                scene_struct['directions'][
                    'front'][1]

            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < min_dist:
                    print((xx, yy, rr))
                    print((x, y, r))
                    print('dist is ', dist)
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        print(x, xx)
                        print(y, yy)
                        print(margin, args.margin, direction_name)
                        print('BROKEN MARGIN!')
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        # Choose color and shape
        if shape_color_combos is None:
            # obj_name, obj_name_out = random.choice(object_mapping)
            obj_name_out = specified_obj.object_type
            obj_name = object_mapping[obj_name_out]
            print(obj_name, obj_name_out)
            color_name = specified_obj.attributes['color'].attr_val
            rgba = color_name_to_rgba[color_name]
        else:
            obj_name_out, color_choices = random.choice(shape_color_combos)
            color_name = random.choice(color_choices)
            obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
            rgba = color_name_to_rgba[color_name]

        # For cube, adjust the size a bit, and make rotate it to make it face forward
        if obj_name_out == 'cube':
            # r /= math.sqrt(2)
            theta = 45
        else:
            theta = 0

        # If inside configuration exists in the sample, store the location for the next object
        # This is used to put the object at the same location, but inside the current object
        if put_obj_inside and stored_location is None:
            stored_location = (x, y)

        # Actually add the object to the scene
        obj_name = utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta, stored_location=stored_location, put_obj_inside=put_obj_inside, allow_floating=args.allow_floating_objects)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Attach a random material
        mat_name_out = specified_obj.attributes['material'].attr_val
        mat_name = material_mapping[mat_name_out]
        # mat_name, mat_name_out = random.choice(material_mapping)

        print(mat_name, mat_name_out)
        utils.add_material(mat_name, Color=rgba)

        # Assign block id to material for binvox to work its magic
        object_id = specified_obj.get_block_id()
        bpy.data.objects[obj_name].active_material.name = 'blockid_' + str(object_id)

        # Record data about the object in the scene data structure
        pixel_coords_lefttop, pixel_coords_rightbottom = get_bbox(args, camera, scene_struct, obj.location, obj_name_out, r)

        # guarantee that objects are all in the image
        if not put_obj_inside:
            if pixel_coords_lefttop[0] < 0 or pixel_coords_lefttop[1] < 0 or pixel_coords_rightbottom[0] >= args.width or \
                    pixel_coords_rightbottom[1] >= args.height:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_objects_from_tree(scene_struct, args, camera, tree_max_level)

            # remove objects that are too small
            if not is_valid_bbox(pixel_coords_lefttop, pixel_coords_rightbottom, size_threshold=args.min_obj_2d_size):
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_objects_from_tree(scene_struct, args, camera, tree_max_level)

        specified_obj.bbox = (pixel_coords_lefttop, pixel_coords_rightbottom)
        objects.append({
            'obj_id': 'blockid_' + str(object_id),
            'obj_name': obj_name,
            'shape': obj_name_out,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'rotation': theta,
            'pixel_coords_lefttop': pixel_coords_lefttop,
            'pixel_coords_rightbottom': pixel_coords_rightbottom,
            'color': color_name,
            'bbox_2d': {},
        })

    # Check that all objects are at least partially visible in the rendered image
    all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
    if not all_visible and not put_obj_inside:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print('Some objects are occluded; replacing objects')
        for obj in blender_objects:
            utils.delete_object(obj)
        print('-'*300 + 'VISIBILITY' + '-'*300)
        return add_objects_from_tree(scene_struct, args, camera, tree_max_level)

    # with open("test_chosen.txt", "a") as myfile:
    #     myfile.write(size_name + '\n')

    # In case of 1 object, add extra objects to center the first object. Later make the new ones invisible in voxels
    if len(specified_objects) == 1:
        x_extra, y_extra, r_extra = positions[0]
        r_extra = 0.2
        theta_extra = 0
        rand_id = np.random.randint(1,255)
        while rand_id == object_id:
            rand_id = np.random.randint(1,255)

        offset_extra = np.random.uniform(2,4)
        obj_name_1 = utils.add_object(args.shape_dir, 'Sphere', r_extra, (x_extra + offset_extra, y_extra + offset_extra), theta=theta_extra, put_obj_inside=put_obj_inside)
        utils.add_material(mat_name, Color=rgba)
        bpy.data.objects[obj_name_1].active_material.name = 'blockid_' + str(rand_id)

        offset_extra = np.random.uniform(2,4)
        obj_name_2 = utils.add_object(args.shape_dir, 'Sphere', r_extra, (x_extra - offset_extra, y_extra - offset_extra), theta=theta_extra, put_obj_inside=put_obj_inside)
        utils.add_material(mat_name, Color=rgba)
        bpy.data.objects[obj_name_2].active_material.name = 'blockid_' + str(rand_id)

    return objects, blender_objects, tree


def get_bbox(args, camera, scene_struct, obj_loc, obj_type, r):
    if obj_type == 'sphere':
        points_3d = [obj_loc + r * vector for vector in get_sphere_unit_vectors(scene_struct['directions'])]
    elif obj_type == 'cube':
        points_3d = [obj_loc + r * (
                Vector(scene_struct['directions']['below']) + Vector(scene_struct['directions']['left']) + Vector(
                    scene_struct['directions']['front'])),
                     obj_loc + r * (Vector(scene_struct['directions']['below']) + Vector(
                         scene_struct['directions']['right']) + Vector(scene_struct['directions']['front'])),
                     obj_loc + r * (Vector(scene_struct['directions']['above']) + Vector(
                         scene_struct['directions']['left']) + Vector(scene_struct['directions']['front'])),
                     obj_loc + r * (Vector(scene_struct['directions']['above']) + Vector(
                         scene_struct['directions']['right']) + Vector(scene_struct['directions']['front'])),
                     obj_loc + r * (Vector(scene_struct['directions']['below']) + Vector(
                         scene_struct['directions']['left']) + Vector(scene_struct['directions']['behind'])),
                     obj_loc + r * (Vector(scene_struct['directions']['below']) + Vector(
                         scene_struct['directions']['right']) + Vector(scene_struct['directions']['behind'])),
                     obj_loc + r * (Vector(scene_struct['directions']['above']) + Vector(
                         scene_struct['directions']['left']) + Vector(scene_struct['directions']['behind'])),
                     obj_loc + r * (Vector(scene_struct['directions']['above']) + Vector(
                         scene_struct['directions']['right']) + Vector(scene_struct['directions']['behind']))
                     ]
    elif obj_type == 'cylinder':
        points_3d = [obj_loc + r * vector for vector in get_cylinder_unit_vectors(scene_struct['directions'])]
    elif obj_type == 'cup':
        # Copied from cylinder's 3d point calculation. Replace later
        points_3d = [obj_loc + r * vector for vector in get_cylinder_unit_vectors(scene_struct['directions'])]
    else:
        raise RuntimeError('invalid object type name')

    points_2d = [utils.get_camera_coords(camera, location) for location in points_3d]
    x_cords = [location[0] for location in points_2d]
    y_cords = [location[1] for location in points_2d]
    left_top = (min(x_cords), min(y_cords))
    right_bottom = (max(x_cords), max(y_cords))

    return left_top, right_bottom

def obj_centered_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return x, y, z

def get_cylinder_unit_vectors(directions):
    points_3d = list()
    for i in range(30):
        theta = (2 * i * math.pi) / 30
        points_3d.append(
            math.cos(theta) * Vector(directions['right']) + math.sin(theta) * Vector(directions['front']) + Vector(
                directions['above']))
        points_3d.append(
            math.cos(theta) * Vector(directions['right']) + math.sin(theta) * Vector(directions['front']) + Vector(
                directions['below']))
    return points_3d


def get_sphere_unit_vectors(directions):
    points_3d = list()
    for i in range(30):
        alpha = i * math.pi / 30 - math.pi / 2  # range in (-pi/2, pi/2)
        for j in range(30):
            theta = (2 * j * math.pi) / 30  # range in (0, 2*pi)
            points_3d.append(
                math.cos(alpha) * math.cos(theta) * Vector(directions['right']) + math.cos(alpha) * math.sin(theta) *
                Vector(directions['front']) + math.sin(alpha) * Vector(directions['above']))

    return points_3d

def get_cup_unit_vectors(directions):
    points_3d = list()
    for i in range(30):
        alpha = i * math.pi / 30 - math.pi / 2  # range in (-pi/2, pi/2)
        for j in range(30):
            theta = (2 * j * math.pi) / 30  # range in (0, 2*pi)
            points_3d.append(
                math.cos(alpha) * math.cos(theta) * Vector(directions['right']) + math.cos(alpha) * math.sin(theta) *
                Vector(directions['front']) + math.sin(alpha) * Vector(directions['above']))

    return points_3d

def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below': continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2: continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def is_valid_bbox(left_top_coord, right_bottom_coord, size_threshold):
    width = right_bottom_coord[0] - left_top_coord[0]
    height = right_bottom_coord[1] - left_top_coord[1]
    if min(width, height) >= size_threshold:
        return True
    else:
        return False


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix='.png')
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter((p[i], p[i + 1], p[i + 2], p[i + 3])
                          for i in range(0, len(p), 4))
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors: break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    return object_colors


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')

    # import pickle
    #
    # with open('../output/trees/train/CLEVR_new_000000.tree', 'rb') as f:
    #     data = pickle.load(f)
    # print(data.function_obj)
