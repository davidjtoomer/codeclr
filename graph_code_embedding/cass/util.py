from typing import List

import torch
import torchtext

from .cass import CassConfig, CassNode, CassTree, NodeType
from .. import DenseGraph


def load_file(file_name, config: CassConfig = None):
    casses = []
    with open(file_name) as f:
        for line in f:
            cass = deserialize(line, config)
            if cass is not None:
                casses.append(cass)
    return casses


def deserialize(s, config: CassConfig):
    tokens = s.strip().split('\t')
    return deserialize_from_tokens(tokens, config)


def deserialize_from_tokens(tokens, config: CassConfig = None):
    num_tokens = len(tokens)
    if num_tokens == 0:
        return None

    num_nodes = int(tokens[0])

    nodes = []
    leaf_nodes = []

    i = 1

    has_fun_sig = False
    if tokens[i][0] == 'S':
        has_fun_sig = True
        fun_sig = tokens[i]
        i += 1
        fun_sig = fun_sig[1:]
        nodes.append(CassNode(NodeType.FunSig, fun_sig))

    while i < num_tokens:
        node_type_label = tokens[i]
        i += 1
        node_type_str = node_type_label[0]
        label = node_type_label[1:]
        if node_type_str == 'I':
            num_child = int(tokens[i])
            i += 1
            nodes.append(
                CassNode(
                    NodeType.Internal,
                    label,
                    [None] *
                    num_child,
                    config=config))
        elif node_type_str == 'N':
            node = CassNode(NodeType.NumLit, label, config=config)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'C':
            node = CassNode(NodeType.CharLit, label, config=config)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'S':
            node = CassNode(NodeType.StringLit, label, config=config)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'V':
            node = CassNode(NodeType.GlobalVar, label, config=config)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'F':
            node = CassNode(NodeType.GlobalFun, label, config=config)
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'v':
            prev_use = int(tokens[i])
            next_use = int(tokens[i + 1])
            i += 2
            node = CassNode(NodeType.LocalVar, label, config=config)
            node.prev_use = prev_use
            node.next_use = next_use
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'f':
            prev_use = int(tokens[i])
            next_use = int(tokens[i + 1])
            i += 2
            node = CassNode(NodeType.LocalFun, label, config=config)
            node.prev_use = prev_use
            node.next_use = next_use
            nodes.append(node)
            leaf_nodes.append(node)
        elif node_type_str == 'E':
            node = CassNode(NodeType.Error, config=config)
            nodes.append(node)
            leaf_nodes.append(node)
        else:
            raise Exception()

    assert num_nodes == len(nodes)

    for n in nodes:
        if n.node_type == NodeType.LocalVar or n.node_type == NodeType.LocalFun:
            if n.prev_use >= 0:
                n.prev_use = nodes[n.prev_use]
            else:
                n.prev_use = None
            if n.next_use >= 0:
                n.next_use = nodes[n.next_use]
            else:
                n.next_use = None

    if has_fun_sig:
        tree_start = 1
    else:
        tree_start = 0

    root, rem_nodes = build_tree_rec(nodes[tree_start:])

    assert root == nodes[tree_start]
    assert len(rem_nodes) == 0

    return CassTree(nodes, leaf_nodes)


def build_tree_rec(nodes):
    node = nodes[0]
    nodes = nodes[1:]
    for i in range(len(node.children)):
        child, nodes = build_tree_rec(nodes)
        child.parent = node
        child.child_id = i
        node.children[i] = child
    return node, nodes


def cass_tree_to_graph(
        cass_trees: List[CassTree],
        vocabulary: torchtext.vocab.Vocab = None) -> DenseGraph:
    nodes = []
    [nodes.extend(cass_tree.nodes) for cass_tree in cass_trees]
    num_nodes = len(nodes)
    for i, node in enumerate(nodes):
        node.set_id(i)

    node_features = torch.zeros(num_nodes, 2)
    adjacency_matrix = torch.zeros(num_nodes, num_nodes)
    for node in nodes:
        node_features[node.id, 0] = node.node_type.value[0]
        if node.n:
            node_features[node.id, 1] = vocabulary[node.n]
        else:
            node_features[node.id, 1] = vocabulary['']
        for child in node.children:
            adjacency_matrix[node.id, child.id] = 1
            adjacency_matrix[child.id, node.id] = 1
        adjacency_matrix[node.id, node.id] = 1
    return DenseGraph(
        node_features=node_features,
        adjacency_matrix=adjacency_matrix)
