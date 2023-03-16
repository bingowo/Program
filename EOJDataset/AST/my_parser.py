from typing import *

import tree_sitter
class ParseError(Exception):
    pass


def is_punctuation(sent: str) -> bool:
    punctuation_list = """
    '";,{}[]()\n
    """
    punctuation_list = list(punctuation_list)

    return sent in punctuation_list


def is_comment(sent: str) -> bool:
    if sent[0:2] == "//":
        return True
    if sent[0:2] == "/*":
        return True
    return False


def parse(code: str, lang: str = "c"):# -> tree_tools.TreeVE:
    parser = tree_sitter.Parser()
    parser.set_language(tree_sitter.Language("EOJDataset/AST/build/my-languages.so", lang))
    try:
        tree = parser.parse(bytes(code, encoding="utf-8"))
    except:
        raise ParseError
        
    V = list()
    E = (list(), list())
    
    def walk(node: tree_sitter.Node):
        desc = node.type
        if not node.children:
            if node.is_named:
                desc = node.text.decode('utf-8')
            if is_punctuation(desc) or is_comment(desc):
                return None
        # print(desc)
        V.append(desc)
        vid = len(V) - 1
        for child in node.children:
            child_vid = walk(child)
            if child_vid is None:
                continue
            E[0].append(child_vid)
            E[1].append(vid)
        # print(desc)
        return vid
    
    walk(tree.root_node)
    return (V, E)


def parse_to_tensor(code: str, lang: str = "c"):
    pass