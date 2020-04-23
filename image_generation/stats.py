import _init_paths
from collections import Counter
import os
import os.path as osp
import pickle

def get_tree_words(tree):
    words = [tree.word]
    for child in tree.children:
        words += get_tree_words(child)
    return words

train_dir = '/local-scratch/cjc/GenerativeNeuralModuleNetwork/data/CLEVR/CLEVR_64_ZS_R/trees/train'
test_dir = '/local-scratch/cjc/GenerativeNeuralModuleNetwork/data/CLEVR/CLEVR_64_ZS_R/trees/test'

c_train = Counter()

print('train dataset stats:')
for tree_file in sorted(os.listdir(train_dir)):
    tree_path = osp.join(train_dir, tree_file)
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    tree_words = get_tree_words(tree)
    c_train.update(tree_words)
print(c_train)
print(len(c_train))

c_test = Counter()

print('test dataset stats:')
for tree_file in sorted(os.listdir(test_dir)):
    tree_path = osp.join(test_dir, tree_file)
    with open(tree_path, 'rb') as f:
        tree = pickle.load(f)
    tree_words = get_tree_words(tree)
    c_test.update(tree_words)
print(c_test)
print(len(c_test))

#for tree_file in sorted(os.listdir(test_dir)):
#    tree_words = get_tree_words(tree)
#    c.update(tree_words)
#print(c)
