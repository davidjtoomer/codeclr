import os

import pytest

from graph_code_embedding.cass import CassTree, load_file


def test_deserialize_file_one_cass_tree() -> None:
    file_name = os.path.join(
        os.path.dirname(__file__),
        'examples',
        'one_cass_tree.cas')
    cass_trees = load_file(file_name)
    assert(len(cass_trees) == 1)
    assert(isinstance(cass_trees[0], CassTree))


def test_deserialize_file_multiple_cass_trees() -> None:
    file_name = os.path.join(
        os.path.dirname(__file__),
        'examples',
        'multiple_cass_trees.cas')
    cass_trees = load_file(file_name)
    assert(len(cass_trees) == 4)
    for cass_tree in cass_trees:
        assert(isinstance(cass_tree, CassTree))
