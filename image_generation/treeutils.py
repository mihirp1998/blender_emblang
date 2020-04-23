import _init_paths
import os
import sys
import argparse
import os.path as osp
import random
import numpy as np
import utils
from mathutils import Vector
from lib.tree import Tree
from modules import Layout, Combine, Describe

# Made changes to expand_tree(), sample_tree_flexible(), refine_tree_info(), _set_describe_bbox(), _combine_bbox()
# Ready for 3d

######### hyperparameters ##########

# module list
module_list = ['layout', 'describe', 'combine']

# children dict
children_dict = dict()
children_dict['layout'] = 2
children_dict['describe'] = 1
children_dict['combine'] = 1

# we will have two split dict for modules for designing a zero-shot setting

module_dict_split1 = dict()
module_dict_split2 = dict()
module_dict_all = dict()

# Zero shot split
# objects list
module_dict_split1['describe'] = ['cube']
module_dict_split2['describe'] = ['cylinder', 'sphere']
module_dict_all['describe'] = ['cylinder', 'cube', 'sphere']
# module_dict_all['describe'] = ['cup']

# attributes list
attribute_list = ['material', 'color', 'size']

module_dict_split1['combine'] = {'material': ['metal'],
                                 'color': ['green', 'blue', 'yellow', 'red'],
                                 'size': ['large', 'small']}

module_dict_split2['combine'] = {'material': ['rubber'],
                                 'color': ['cyan', 'brown', 'gray', 'purple'],
                                 'size': ['small', 'large']}

module_dict_all['combine'] = {'material': ['rubber', 'metal'],
                              'color': ['cyan', 'brown', 'gray', 'purple', 'green', 'blue', 'yellow', 'red'],
                              'size': ['small', 'large']}
# relations list
module_dict_split1['layout'] = ['left', 'left-front', 'right-front']
module_dict_split2['layout'] = ['right', 'right-behind', 'left-behind']
module_dict_all['layout'] = ['right', 'left', 'right-behind', 'left-front', 'left-behind',
                             'right-front', 'front', 'behind']

module_dicts_zeroshot = [module_dict_split1, module_dict_split2]
module_dict_normal = module_dict_all


module_dicts_inside = dict()
module_dicts_inside['describe'] = ['cylinder', 'cube', 'sphere', 'cup']
module_dicts_inside['combine'] = {'material': ['rubber', 'metal'],
                              'color': ['cyan', 'brown', 'gray', 'purple', 'green', 'blue', 'yellow', 'red'],
                              'size': ['large']}
module_dicts_inside['layout'] = ['inside']


module_dicts_back_front = dict()
module_dicts_back_front['describe'] = ['cylinder', 'cube', 'sphere']
module_dicts_back_front['combine'] = {'material': ['rubber', 'metal'],
                              'color': ['cyan', 'brown', 'gray', 'purple', 'green', 'blue', 'yellow', 'red'],
                              'size': ['large']}
module_dicts_back_front['layout'] = ['front', 'behind']


pattern_map = {'describe': 0, 'material': 1, 'color': 2, 'size': 3, 'layout': 4}

zs_training_patterns = [(0, 1, 0, 1, 0), (1, 0, 1, 0, 1)]
zs_training_probs = [1.0 / 3, 2.0 / 3]
zs_test_patterns = [(1, 1, 1, 1, 1), (0, 0, 0, 0, 0), (0, 0, 1, 1, 1), (1, 1, 0, 0, 0), (0, 1, 1, 1, 0),
                    (1, 0, 0, 0, 1), (0, 1, 1, 1, 1), (1, 0, 0, 0, 0)]
zs_test_probs = [1.0 / 6, 1.0 / 12, 1.0 / 12, 1.0 / 6, 1.0 / 12, 1.0 / 6, 1.0 / 12, 1.0 / 6]


def expand_tree_with_inside(tree, level, parent, memorylist, child_idx, max_layout_level, add_layout_prob, train, obj_count, zero_shot=False,
                metadata_pattern=None, back_front_only_flag=False):
    if parent is None or parent.function == 'layout':
        # sample module, the module can be either layout or describe here
        if level + 1 > max_layout_level:
            module_idx = 1
        else:
            module_idx = 0
        tree.function = module_list[module_idx]
        if zero_shot and (level == 0 or tree.function == 'describe'):
            r = random.random()
            if train:
                metadata_pattern = _choose_pattern(zs_training_patterns, zs_training_probs, r)
            else:
                metadata_pattern = _choose_pattern(zs_test_patterns, zs_test_probs, r)
        # sample content
        if zero_shot:
            assert (metadata_pattern is not None)
            dict_index = metadata_pattern[pattern_map[tree.function]]
            module_dict = module_dicts_zeroshot[dict_index]
        else:
            module_dict = module_dicts_inside

        if tree.function == 'describe' and child_idx == 1:
            tree.word = module_dict[tree.function][-1]
        elif tree.function == 'layout':
            tree.word = module_dict[tree.function][-1]
        else:
            word_id = random.randint(0, len(module_dict[tree.function]) - 2)
            tree.word = module_dict[tree.function][word_id]

        if tree.function == 'layout':
            tree.function_obj = Layout(tree.word)
            # print('add layout')
        else:
            obj_count += 1
            tree.function_obj = Describe(tree.word, obj_count)
            # print('add describe')

        tree.num_children = children_dict[tree.function]
        if parent is not None:  # then the parent must be a layout node
            if child_idx == 0:
                parent.function_obj.left_child = tree.function_obj
            else:
                parent.function_obj.right_child = tree.function_obj

        for i in range(tree.num_children):
            tree.children.append(Tree())
            tree.children[i], obj_count = expand_tree_with_inside(tree.children[i], level + 1, tree, [], i, max_layout_level,
                                           add_layout_prob,
                                           train, obj_count, zero_shot, metadata_pattern, back_front_only_flag)

    # must contain only one child node, which is a combine node
    elif parent.function == 'describe' or parent.function == 'combine':
        # print('add combine')
        valid = [2]
        # no need to sample module for now
        module_id = 0
        tree.function = module_list[valid[module_id]]

        # sample content
        # sample which attributes
        if len(set(attribute_list) - set(memorylist)) <= 1:
            full_attribute = True
        else:
            full_attribute = False

        attribute = random.sample(set(attribute_list) - set(memorylist), 1)[0]
        memorylist += [attribute]

        if zero_shot:
            assert (metadata_pattern is not None)
            dict_idx = metadata_pattern[pattern_map[attribute]]
            module_dict = module_dicts_zeroshot[dict_idx]
        else:
            module_dict = module_dict_normal

        word_id = random.randint(0, len(module_dict[tree.function][attribute]) - 1)
        tree.word = module_dict[tree.function][attribute][word_id]

        if isinstance(parent.function_obj, Describe):
            carrier = parent.function_obj
        else:
            carrier = parent.function_obj.get_carrier()

        tree.function_obj = Combine(attribute, tree.word)
        tree.function_obj.set_carrier(carrier)
        carrier.set_attribute(attribute, tree.function_obj)

        if not full_attribute:
            tree.num_children = children_dict[tree.function]

            for i in range(tree.num_children):
                tree.children.append(Tree())
                tree.children[i], obj_count = expand_tree_with_inside(tree.children[i], level + 1, tree, memorylist, i, max_layout_level,
                                               add_layout_prob,
                                               train, obj_count, zero_shot, metadata_pattern, back_front_only_flag)
    else:
        raise ValueError('Wrong function.')
    return tree, obj_count



def expand_tree(tree, level, parent, memorylist, child_idx, max_layout_level, add_layout_prob, train, obj_count, zero_shot=False,
                metadata_pattern=None, back_front_only_flag=False):
    if parent is None or parent.function == 'layout':
        # sample module, the module can be either layout or describe here
        if level + 1 > max_layout_level:
            module_idx = 1
        else:
            rand = random.random()
            if rand >= 1 - add_layout_prob:
                module_idx = 0
            else:
                module_idx = 1
        tree.function = module_list[module_idx]
        if zero_shot and (level == 0 or tree.function == 'describe'):
            r = random.random()
            if train:
                metadata_pattern = _choose_pattern(zs_training_patterns, zs_training_probs, r)
            else:
                metadata_pattern = _choose_pattern(zs_test_patterns, zs_test_probs, r)
        # sample content
        if zero_shot:
            assert (metadata_pattern is not None)
            dict_index = metadata_pattern[pattern_map[tree.function]]
            module_dict = module_dicts_zeroshot[dict_index]
        elif back_front_only_flag:
            module_dict = module_dicts_back_front
        else:
            module_dict = module_dict_normal

        word_id = random.randint(0, len(module_dict[tree.function]) - 1)
        tree.word = module_dict[tree.function][word_id]

        if tree.function == 'layout':
            tree.function_obj = Layout(tree.word)
            # print('add layout')
        else:
            obj_count += 1
            tree.function_obj = Describe(tree.word, obj_count)
            # print('add describe')

        tree.num_children = children_dict[tree.function]
        if parent is not None:  # then the parent must be a layout node
            if child_idx == 0:
                parent.function_obj.left_child = tree.function_obj
            else:
                parent.function_obj.right_child = tree.function_obj

        for i in range(tree.num_children):
            tree.children.append(Tree())
            tree.children[i], obj_count = expand_tree(tree.children[i], level + 1, tree, [], i, max_layout_level,
                                           add_layout_prob,
                                           train, obj_count, zero_shot, metadata_pattern, back_front_only_flag)

    # must contain only one child node, which is a combine node
    elif parent.function == 'describe' or parent.function == 'combine':
        # print('add combine')
        valid = [2]
        # no need to sample module for now
        module_id = 0
        tree.function = module_list[valid[module_id]]

        # sample content
        # sample which attributes
        if len(set(attribute_list) - set(memorylist)) <= 1:
            full_attribute = True
        else:
            full_attribute = False

        attribute = random.sample(set(attribute_list) - set(memorylist), 1)[0]
        memorylist += [attribute]

        if zero_shot:
            assert (metadata_pattern is not None)
            dict_idx = metadata_pattern[pattern_map[attribute]]
            module_dict = module_dicts_zeroshot[dict_idx]
        else:
            module_dict = module_dict_normal

        word_id = random.randint(0, len(module_dict[tree.function][attribute]) - 1)
        tree.word = module_dict[tree.function][attribute][word_id]

        if isinstance(parent.function_obj, Describe):
            carrier = parent.function_obj
        else:
            carrier = parent.function_obj.get_carrier()

        tree.function_obj = Combine(attribute, tree.word)
        tree.function_obj.set_carrier(carrier)
        carrier.set_attribute(attribute, tree.function_obj)

        if not full_attribute:
            tree.num_children = children_dict[tree.function]

            for i in range(tree.num_children):
                tree.children.append(Tree())
                tree.children[i], obj_count = expand_tree(tree.children[i], level + 1, tree, memorylist, i, max_layout_level,
                                               add_layout_prob,
                                               train, obj_count, zero_shot, metadata_pattern, back_front_only_flag)
    else:
        raise ValueError('Wrong function.')
    return tree, obj_count


def _choose_pattern(patterns, probs, r):
    assert (sum(probs) == 1, 'Given prob list should sum up to 1')
    assert (len(patterns) == len(probs), 'Given patterns should have the same length as the given probs')
    accum = 0
    for i, prob in enumerate(probs):
        accum += prob
        if r < accum:
            return patterns[i]


def visualize_trees(trees):
    for i in range(len(trees)):
        print('************** tree **************')
        _visualize_tree(trees[i], 0)
        print('**********************************')


def _visualize_tree(tree, level):
    if tree == None:
        return
    for i in range(tree.num_children - 1, (tree.num_children - 1) // 2, -1):
        _visualize_tree(tree.children[i], level + 1)

    print(' ' * level + tree.word)
    if isinstance(tree.function_obj, Describe):
        print(tree.function_obj.attributes, tree.function_obj)
        if tree.function != 'combine':
            print('position {}'.format(tree.function_obj.position))

    for i in range((tree.num_children - 1) // 2, -1, -1):
        _visualize_tree(tree.children[i], level + 1)

    return


def allign_tree(tree, level):
    """
        A pre-order traversal, set the position of tree nodes according to the layouts
    :param tree:
    :return:
    """
    if tree is None:
        return

    if tree.function == 'describe' and level == 0:
        tree.function_obj.set_random_pos()
    elif tree.function == 'layout':
        tree.function_obj.set_children_pos()
        for i in range(tree.num_children):
            allign_tree(tree.children[i], level + 1)
    else:
        pass


def extract_objects(tree):
    objects = list()

    if tree is None:
        return objects

    if tree.function == 'describe':
        objects.append(tree.function_obj)
    elif tree.function == 'layout':
        for i in range(tree.num_children):
            objects += extract_objects(tree.children[i])
    else:
        pass

    return objects


def sample_tree(max_layout_level, add_layout_prob, obj_count, zero_shot=False, train=True):
    tree = Tree()
    tree, obj_count = expand_tree(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, obj_count, zero_shot=zero_shot)
    allign_tree(tree, 0)
    return tree


def sample_tree_flexible(percent_inside_samples, include_inside_config, max_layout_level, add_layout_prob, obj_count, zero_shot=False, train=True, arguments=None, back_front_only_flag=False):
    tree = Tree()

    if not include_inside_config:
        expand_func = expand_tree
    else:
        rand = random.random()
        if rand < percent_inside_samples:
            arguments = {'fix_num_objs':2}
            expand_func = expand_tree_with_inside
            max_layout_level = 1
        else:
            expand_func = expand_tree

    if arguments is None:
        tree, obj_count = expand_func(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, obj_count, zero_shot=zero_shot, back_front_only_flag=back_front_only_flag)
    else:
        max_num_objs = arguments['max_num_objs']
        min_num_objs = arguments['min_num_objs']
        object_count_range = range(min_num_objs, max_num_objs + 1)

        tree, obj_count = expand_func(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, obj_count, zero_shot=zero_shot, back_front_only_flag=back_front_only_flag)
        num_objs = count_functions(tree, 'describe')
        while num_objs not in object_count_range:
            tree = Tree()
            tree, obj_count = expand_func(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, obj_count, zero_shot=zero_shot, back_front_only_flag=back_front_only_flag)
            num_objs = count_functions(tree, 'describe')

        # if 'max_num_objs' in arguments:
        #     max_num_objs = arguments['max_num_objs']
        #     tree, obj_count = expand_func(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, obj_count, zero_shot=zero_shot, back_front_only_flag=back_front_only_flag)
        #     num_objs = count_functions(tree, 'describe')
        #     while num_objs > max_num_objs:
        #         tree = Tree()
        #         tree, obj_count = expand_func(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, obj_count, zero_shot=zero_shot, back_front_only_flag=back_front_only_flag)
        #         num_objs = count_functions(tree, 'describe')
        #         print(num_objs)
        # elif 'fix_num_objs' in arguments:
        #     fix_num_objs = arguments['fix_num_objs']
        #     tree, obj_count = expand_func(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, obj_count, zero_shot=zero_shot, back_front_only_flag=back_front_only_flag)
        #     num_objs = count_functions(tree, 'describe')
        #     while num_objs != fix_num_objs:
        #         tree = Tree()
        #         tree, obj_count = expand_func(tree, 0, None, [], 0, max_layout_level, add_layout_prob, train, obj_count, zero_shot=zero_shot, back_front_only_flag=back_front_only_flag)
        #         num_objs = count_functions(tree, 'describe')
    allign_tree(tree, 0)

    return tree


def count_functions(tree, name):
    num_functions = _count_functions(tree, name)
    return num_functions


def _count_functions(tree, name):
    num_objs = 0
    for i in range(0, tree.num_children):
        num_objs += _count_functions(tree.children[i], name)

    if tree.function == name:
        num_objs += 1

    return num_objs

# def _project_bbox(camera, points_3d):
#     points_2d = [utils.get_camera_coords(camera, Vector(location)) for location in points_3d]
#     x_cords = [location[0] for location in points_2d]
#     y_cords = [location[1] for location in points_2d]
#     left_top = (min(x_cords), min(y_cords))
#     right_bottom = (max(x_cords), max(y_cords))
#     # print('-'*50)
#     # print(points_3d, points_2d)
#     # print('-'*50)
#     return [left_top, right_bottom]

# def _get_describe_bbox(tree, blocks, camera, bboxes_2d):
#     function_obj = tree.function_obj
#     # set the bbox for the tree node
#     if hasattr(function_obj, 'bbox'):
#         block_id = function_obj.block_id
#         object_idx = np.where(blocks == block_id)
#         x_top = object_idx[0].min()
#         y_top = object_idx[1].min()
#         z_top = object_idx[2].min()

#         x_bottom = object_idx[0].max()
#         y_bottom = object_idx[1].max()
#         z_bottom = object_idx[2].max()

#         points_3d = [[x_top, y_top, z_top],[x_bottom - x_top,
#                 y_bottom - y_top, z_bottom - z_top]]
#         bbox = _project_bbox(camera, points_3d)
#         bboxes_2d[block_id] = bbox

#     for child in tree.children:
#         bboxes_2d = _get_describe_bbox(child, blocks, camera, bboxes_2d)
#     return bboxes_2d

# def get_2d_bboxes(tree, blocks, camera):
#     bboxes_2d = {}
#     bboxes_2d = _get_describe_bbox(tree, blocks, camera, bboxes_2d)
#     return bboxes_2d

def refine_tree_info(tree, blocks):
    tree = _set_describe_bbox(tree, blocks)
    tree = _set_layout_bbox(tree)
    return tree


def remove_function_obj(tree):
    tree = _remove_function_obj(tree)
    return tree


def _remove_function_obj(tree):
    if hasattr(tree, 'function_obj'):
        delattr(tree, 'function_obj')
    for child in tree.children:
        _remove_function_obj(child)
    return tree


def _set_describe_bbox(tree, blocks):
    function_obj = tree.function_obj
    # set the bbox for the tree node
    if hasattr(function_obj, 'bbox'):
        block_id = function_obj.block_id
        object_idx = np.where(blocks == block_id)
        # try:
        #     x_top = object_idx[0].min()
        # except Exception as e:
        #     print(blocks.max())
        #     print(blocks.min())
        #     print(np.unique(blocks))
        #     print(block_id)
        x_top = object_idx[0].min()
        y_top = object_idx[1].min()
        z_top = object_idx[2].min()

        x_bottom = object_idx[0].max()
        y_bottom = object_idx[1].max()
        z_bottom = object_idx[2].max()

        # bbox = (x_top, y_top, z_top, x_bottom - x_top, y_bottom - y_top, z_bottom - z_top)
        bbox = (x_top, z_top, y_top, x_bottom - x_top, z_bottom - z_top, y_bottom - y_top)
        tree.bbox = np.array(bbox)
        # print(x_top, y_top, z_top, x_bottom, y_bottom, z_bottom)
        # print(tree.bbox)
        # print(tree.word)
        # print('--------------')

    for child in tree.children:
        _set_describe_bbox(child, blocks)
    return tree


def _set_layout_bbox(tree):
    if tree.function != 'layout':
        return tree
    else:
        for child in tree.children:
            _set_layout_bbox(child)
        # set the bbox for layout module
        left_child_bbox = tree.children[0].bbox
        right_child_bbox = tree.children[1].bbox
        tree.bbox = np.array(_combine_bbox(left_child_bbox, right_child_bbox))

        return tree


def _correct_layout_word(tree):
    if tree.function != 'layout':
        return tree
    else:
        left_child_bbox = tree.children[0].bbox
        right_child_bbox = tree.children[1].bbox
        if left_child_bbox[0] < right_child_bbox[0]:
            if right_child_bbox[1] - 5 < left_child_bbox[1] < right_child_bbox[1] + 5:
                tree.word = 'left'
            elif left_child_bbox[1] <= right_child_bbox[1] - 5:
                tree.word = 'left-behind'
            else:
                tree.word = 'left-front'
        else:
            if right_child_bbox[1] - 5 < left_child_bbox[1] < right_child_bbox[1] + 5:
                tree.word = 'right'
            elif left_child_bbox[1] <= right_child_bbox[1] - 5:
                tree.word = 'right-behind'
            else:
                tree.word = 'right-front'

        for child in tree.children:
            _correct_layout_word(child)

        return tree


def _combine_bbox(bbox1, bbox2):
    x = min(bbox1[0], bbox2[0])
    y = min(bbox1[1], bbox2[1])
    z = min(bbox1[2], bbox2[2])
    x_bottom = max(bbox1[0] + bbox1[3], bbox2[0] + bbox2[3])
    y_bottom = max(bbox1[1] + bbox1[4], bbox2[1] + bbox2[4])
    z_bottom = max(bbox1[2] + bbox1[5], bbox2[2] + bbox2[5])
    return [x, y, z, x_bottom - x, y_bottom - y, z_bottom - z]


def add_parent(tree):
  tree = _add_parent(tree, None)

  return tree

def _add_parent(tree, parent):
  tree.parent = parent
  for i in range(0, tree.num_children):
    tree.children[i] = _add_parent(tree.children[i], tree) 

if __name__ == '__main__':
    # random.seed(12113)
    #
    # # tree = Tree()
    # # tree = expand_tree(tree, 0, None, [], 0)
    # # allign_tree(tree)
    #
    # num_sample = 1
    # trees = []
    # for i in range(num_sample):
    #     treei = Tree()
    #     treei = expand_tree(treei, 0, None, [], 0, max_level=2)
    #     allign_tree(treei, 0)
    #     objects = extract_objects(treei)
    #     trees += [treei]
    #     print(objects)
    #
    # visualize_tree(trees)

    for i in range(1):
        print('normal sample tree')
        tree = sample_tree(max_layout_level=2, add_layout_prob=0.6, zero_shot=True, train=True)
        visualize_trees([tree])
        print('max sample tree')
        tree = sample_tree_flexible(max_layout_level=3, add_layout_prob=0.6, zero_shot=False, train=True,
                                    arguments={'max_num_objs': 3})
        visualize_trees([tree])
        print('fix sample tree')
        tree = sample_tree_flexible(max_layout_level=3, add_layout_prob=0.6, zero_shot=False, train=True,
                                    arguments={'fix_num_objs': 8})
        visualize_trees([tree])
