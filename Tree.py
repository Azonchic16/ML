class Node():

    def __init__(self, value=None):
        self.df = None
        self.parent = None
        self.value = value
        self.left = None
        self.right = None
        self.is_leaf = False


class Tree():

    def __init__(self):
        self.root = None

    def print_tree(self):
        """Печатает дерево в консоли, начиная с корня"""
        if not self.root:
            print("(пустое дерево)")
            return
        
        def _print(node, prefix="", is_left=True):
            if not node:
                return
            
            if node.is_leaf:
                label = f"{node.value[0]} = {node.value[1]}" if is_left else f"{node.value[0]} = {node.value[1]}"
            else:
                label = f"{node.value[0]} > {node.value[1]}"
            
            print(prefix + ("└── " if not prefix else "├── ") + label)
            
            new_prefix = prefix + ("    " if not prefix else "│   ")
            
            _print(node.left, new_prefix, True)
            _print(node.right, new_prefix, False)
        
        _print(self.root, "", False)
        
    def find_proba(self, s):
        def tree_traversal(node, s):
            if node.is_leaf:
                return node.value[1]
            else:
                feature, split_val = node.value[0], node.value[1]
                if s[feature] <= split_val:
                    return tree_traversal(node.left, s)
                else:
                    return tree_traversal(node.right, s)
        return tree_traversal(self.root, s)