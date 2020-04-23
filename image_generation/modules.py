# Made changes to Describe()
# Ready for 3d


import random


def position_add(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return x1 + x2, y1 + y2


def position_mul(pos, scalar):
    x, y = pos
    return x * scalar, y * scalar


class Layout(object):
    # set the bounds for object positions
    random_lower_bound = 0.8
    random_upper_bound = 1.6
    position_dict = {
        'left': ((-1.5, 0), (1.5, 0)),
        'right': ((1.5, 0), (-1.5, 0)),
        'left-front': [((-1.0, 2.0), (1.0, -2.0))],
        'right-front': [((1.0, 2.0), (-1.0, -2.0))],
        'left-behind': [((-1.0, -2.0), (1.0, 2.0))],
        'right-behind': [((1.0, -2.0), (-1.0, 2.0))],
        'inside': ((-1.5, 0), (1.5, 0)),
        'front': ((0, 1.75), (0, -1.75)),
        'behind': ((0, -1.75), (0, 1.75))
    }

    def __init__(self, layout_type):
        self.layout_type = layout_type
        self.position = (0, 0)
        self.left_child = None
        self.right_child = None

    def change_position(self, pos):
        self.position = position_add(self.position, pos)

    def set_children_pos(self):
        layout_setting = self.position_dict[self.layout_type]
        if isinstance(layout_setting, list):
            layout_setting = random.sample(layout_setting, 1)[0]

        scaling = random.uniform(self.random_lower_bound, self.random_upper_bound)
        self.left_child.change_position(position_add(self.position, position_mul(layout_setting[0], scaling)))
        self.right_child.change_position(position_add(self.position, position_mul(layout_setting[1], scaling)))


class Describe(object):
    random_pos_range = 1.5

    def __init__(self, object_type, obj_count):
        self.object_type = object_type
        self.attributes = dict()
        self.position = (0, 0)  # position: tuple of (x,y)
        self.size = 0
        self.bbox = None  # 2D bbox coordinates
        self.block_id = obj_count

    def __repr__(self):
        return '<Describe, obj_type:{}>'.format(self.object_type)

    def set_attribute(self, name, obj):
        self.attributes[name] = obj

    def get_attribute(self, name):
        return self.attributes[name]

    def get_block_id(self):
        return self.block_id

    def change_position(self, pos):
        self.position = position_add(self.position, pos)

    def set_random_pos(self):
        x = random.uniform(-self.random_pos_range, self.random_pos_range)
        y = random.uniform(-self.random_pos_range, self.random_pos_range)
        self.position = (x, y)


class Combine(object):
    def __init__(self, attr_type, val):
        self.carrier = None
        self.attr_type = attr_type
        self.attr_val = val

    def __repr__(self):
        return '<Combine, attr {}:{}>'.format(self.attr_type, self.attr_val)

    def set_carrier(self, obj):
        self.carrier = obj

    def get_carrier(self):
        if self.carrier is None:
            raise ValueError("The Combine module doesn't have carrier!")
        return self.carrier
