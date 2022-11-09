'''
MIT License
Copyright (c) 2021 Intel Labs
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from enum import Enum
from typing import List


class NodeType(Enum):
    Internal = 0,
    NumLit = 1,
    CharLit = 2,
    StringLit = 3,
    GlobalVar = 4,
    GlobalFun = 5,
    LocalVar = 6,
    LocalFun = 7,
    FunSig = 8,
    Error = 9


class CassConfig:
    def __init__(self, annot_mode: int = 0, compound_mode: int = 0, gvar_mode: int = 0, gfun_mode: int = 0, fsig_mode: int = 0):
        self.annot_mode = annot_mode
        self.compound_mode = compound_mode
        self.gfun_mode = gfun_mode
        self.gvar_mode = gvar_mode
        self.fsig_mode = fsig_mode


class CassNode:
    def __init__(self, node_type: NodeType, label: str = '', children: List = [], config: CassConfig = None):
        self.node_type = node_type
        self.children = children
        self.prev_use = None
        self.next_use = None
        self.parent = None
        self.child_id = 0
        self.config = CassConfig() if not config else config

        self.removed = False

        if len(label) == 0:
            self.label = label
            self.n = label

        elif node_type == NodeType.FunSig:
            if self.config.fsig_mode == 0:
                self.n = None
            else:
                self.n = label

        elif node_type == NodeType.Internal:
            assert label[0] == '#'
            p = label[1:].find('#')
            assert p > 0
            p += 2
            self.annot = label[:p]
            self.label = label[p:]

            if self.annot == '#compound_statement#':
                if self.config.compound_mode == 0:
                    pass
                elif self.config.compound_mode == 1:
                    self.removed = True
                elif self.config.compound_mode == 2:
                    self.label = '{#}'
                else:
                    raise Exception()

            if self.config.annot_mode == 0:
                self.n = self.label
            elif self.config.annot_mode == 1:
                self.n = self.annot + self.label
            elif self.config.annot_mode == 2:
                if self.annot == '#parenthesized_expression#' or self.annot == '#argument_list#':
                    self.n = self.annot + self.label
                else:
                    self.n = self.label
            else:
                raise Exception()

        else:
            if node_type == NodeType.LocalVar or node_type == NodeType.LocalFun:
                self.n = '$VAR'

            elif node_type == NodeType.GlobalVar:
                if self.config.gvar_mode == 0:
                    self.n = label
                elif self.config.gvar_mode == 1:
                    self.n = label
                    self.removed = True
                elif self.config.gvar_mode == 2:
                    self.n = '$GVAR'
                elif self.config.gvar_mode == 3:
                    self.n = '$VAR'
                else:
                    raise Exception()

            elif node_type == NodeType.GlobalFun:
                if self.config.gfun_mode == 0:
                    self.n = label
                elif self.config.gfun_mode == 1:
                    self.n = label
                    self.removed = True
                elif self.config.gfun_mode == 2:
                    self.n = '$GFUN'
                elif self.config.gfun_mode == 3:
                    if self.config.gvar_mode == 3:
                        self.n = '$VAR'
                    else:
                        self.n = '$GVAR'
                else:
                    raise Exception()

            else:
                self.n = label

        self.features = []

    def set_id(self, id: int):
        self.id = id


class CassTree:
    def __init__(self, nodes, leaf_nodes):
        self.nodes = nodes
        self.leaf_nodes = leaf_nodes
        if nodes[0].node_type == NodeType.FunSig:
            self.fun_sig_node = nodes[0]
            self.root = nodes[1]
        else:
            self.fun_sig_node = None
            self.root = nodes[0]
        self.leaf_ranges = self._compute_leaf_ranges()

    def _compute_leaf_ranges(self):
        node2leaf_id = {}
        leaf_ranges = {}
        for i, node in enumerate(self.leaf_nodes):
            node2leaf_id[node] = i

        def compute_leaf_ranges_rec(node):
            if len(node.children) == 0:
                x = node2leaf_id[node]
                leaf_ranges[node] = (x, x + 1)
            else:
                for c in node.children:
                    compute_leaf_ranges_rec(c)
                leaf_ranges[node] = (
                    leaf_ranges[node.children[0]][0], leaf_ranges[node.children[-1]][1])

        compute_leaf_ranges_rec(self.root)
        return leaf_ranges

    def _get_context(self, node):
        assert not node.removed

        p = node.parent
        if p is None:
            return None
        if p.label != '$.$':
            if p.removed:
                return None
            return (node.child_id, p.n)
        else:
            for i in range(*(self.leaf_ranges[p])):
                l = self.leaf_nodes[i]
                if l.node_type == NodeType.GlobalVar or l.node_type == NodeType.GlobalFun:
                    if l.removed:
                        return None
                    return l.n
            return None

    def featurize(self):
        for i, node in enumerate(self.leaf_nodes):
            if node.removed:
                continue

            node.features.append(node.n)

            p = node
            for _ in range(3):
                cid = p.child_id
                p = p.parent
                if p is None:
                    break
                if p.removed:
                    continue
                node.features.append((node.n, cid, p.n))

            if i > 0:
                sib = self.leaf_nodes[i - 1]
                if not sib.removed:
                    node.features.append((sib.n, node.n))
            if i < len(self.leaf_nodes) - 1:
                sib = self.leaf_nodes[i + 1]
                if not sib.removed:
                    node.features.append((node.n, sib.n))

            if node.prev_use is not None:
                if not node.prev_use.removed:
                    prev_ctx = self._get_context(node.prev_use)
                    ctx = self._get_context(node)
                    if prev_ctx is not None and ctx is not None:
                        node.features.append((prev_ctx, ctx))
            if node.next_use is not None:
                if not node.next_use.removed:
                    ctx = self._get_context(node)
                    next_ctx = self._get_context(node.next_use)
                    if ctx is not None and next_ctx is not None:
                        node.features.append((ctx, next_ctx))

        features = []
        for n in self.leaf_nodes:
            features += n.features

        if self.config.fsig_mode == 1 and self.fun_sig_node is not None:
            features.append(self.fun_sig_node.n)

        return features
