import bpy
import os

# get the path where the blend file is located
basedir = bpy.path.abspath('./')

# open blender scene
fname = 'CLEVR_new_000001'
inp_blend_file = 'output/blendfiles/train/'+fname+'.blend'
bpy.ops.wm.open_mainfile(filepath=inp_blend_file)


# deselect all objects
bpy.ops.object.select_all(action='DESELECT')    


# for obj in bpy.data.objects:
#     #if the object is a mesh and not a lamp or camera etc.
#     print (obj.name)
#     print (obj.type)
#     print (obj.location)
#     print (obj.active_material.name)
#     print (obj.material_slots)
#     print (obj.bound_box.data.data)
#     print ('--------')
    # print (dir(obj))

i = 1
for obj in bpy.data.objects:
    #if the object is a mesh and not a lamp or camera etc.
    if obj.type == 'MESH' and obj.name != 'Ground':
        obj.select = True
        obj.active_material.name = 'blockid_' + str(i)
        i += 1

# # bpy.ops.export_scene.obj(filepath=os.path.join(basedir, fname + '.obj'), use_mesh_modifiers=True)

bpy.ops.export_scene.obj(
        filepath=os.path.join(basedir, fname + '.obj'),
        use_selection=True,
        use_materials=True,
        use_mesh_modifiers=True,
        )

# /Users/ashar/work/visual_imagination/prob_scene_gen/ProbabilisticNeuralProgrammedNetwork/data/CLEVR/clevr-dataset-gen/image_generation/


# # loop through all the objects in the scene
# scene = bpy.context.scene
# for ob in scene.objects:
#     # make the current object active and select it
#     scene.objects.active = ob
#     ob.select = True

#     # make sure that we only export meshes
#     if ob.type == 'MESH':
#         # export the currently selected object to its own file based on its name
#         bpy.ops.export_scene.obj(
#                 filepath=os.path.join(basedir, ob.name + '.obj'),
#                 use_selection=True,
#                 )
#     # deselect the object and move on to another if any more are left
#     ob.select = False