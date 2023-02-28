from __future__ import print_function

from pycparser.c_ast import *

from remove_zs import rm_emptyline, rm_includeline, rmCommentsInCFile

sys.path.extend(['.', '..'])

from pycparser import c_parser

from anytree import Node, RenderTree
from anytree.search import findall_by_attr
from anytree.walker import Walker
import random
import os

import pandas as pd
from pycparser import parse_file

# 读取c文件程序的内容并去除注释、空行以及头文件，最后生成ast
# 返回内容为处理后的源代码 和 ast
def translate_to_c(filename):
    # 这里展示不用本地编译器的方法
    # 但读取的文本序列，需去除#include #define 以及注释 这类语句才能生成AST
    with open(filename, encoding='utf-8') as f:
        txt = f.read()
    # txt = rmCommentsInCFile(txt)  # 去除注释
    # txt = rm_emptyline(txt)  # 去除空行
    txt = rm_includeline(txt)  # 去除头文件

    txt = '#define bool int\n' + txt
    txt = '#define uint32_t unsigned int\n' + txt
    txt = '#define __int64_t long long\n' + txt
    txt = '#define size_t unsigned long long \n' + txt
    txt = '#define extern int \n' + txt
    txt = '#define int32_t int\n' + txt
    
    
    

    with open('EOJDataset/contest513/tmp.c','w', encoding='utf-8') as f:
        f.write(txt)
    # print(txt)

    # ast = c_parser.CParser().parse(txt)
    # c_parser.CParser.parse()
    # print(ast)

    ast = parse_file('EOJDataset/contest513/tmp.c', use_cpp = True, cpp_path=r'C:\\Program Files (x86)\Dev-Cpp\\MinGW64\bin\\gcc.exe', cpp_args=['-E'])#, r'-Iutils/fake_libc_include'])
    return txt, ast


# 直接将 源代码字符串 转变为 ast
# 返回内容为处理后的源代码 和 ast
def translate_to_c_txt(txt):
    # txt = rmCommentsInCFile(txt)  # 去除注释
    txt = rm_emptyline(txt)  # 去除空行
    txt = rm_includeline(txt)  # 去除头文件
    # print(txt)
    ast = c_parser.CParser().parse(txt)
    # print(ast)
    return txt, ast


def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    else:
        token = node.__class__.__name__

    return token

def get_children(root):

    def expand(nested_list):
        for _, item in nested_list:
            yield item

    children = []
    if isinstance(root, str):
        children = []
    elif len(root.children()) != 0:
        children = expand(root.children())
    elif len(root.attr_names) != 0:
        nodeAttributes = root.attr_names
        for attr in nodeAttributes:
            attribute = getattr(root, attr) # 先获取属性
            if isinstance(attribute, str):
                children.append(attribute)
            elif attribute != None:
                children.extend(attribute) 

    return list(children)

def get_trees(current_node, parent_node, order):
    
    token, children = get_token(current_node), get_children(current_node)
    node = Node([order,token], parent=parent_node, order=order)

    for child_order in range(len(children)):
        get_trees(children[child_order], node, order+str(int(child_order)+1))

def get_path_length(path):
    """Calculating path length.
    Input:
    path: list. Containing full walk path.

    Return:
    int. Length of the path.
    """
    
    return len(path)

def get_path_width(raw_path):
    """Calculating path width.
    Input:
    raw_path: tuple. Containing upstream, parent, downstream of the path.

    Return:
    int. Width of the path.
    """
    
    return abs(int(raw_path[0][-1].order)-int(raw_path[2][0].order))
    
def hashing_path(path, hash_table):
    """Calculating path width.
    Input:
    raw_path: tuple. Containing upstream, parent, downstream of the path.

    Return:
    str. Hash of the path.
    """
    
    if path not in hash_table:
        hash = random.getrandbits(128)
        hash_table[path] = str(hash)
        return str(hash)
    else:
        return hash_table[path]
    
def get_node_rank(node_name, max_depth):
    """Calculating node rank for leaf nodes.
    Input:
    node_name: list. where the first element is the string order of the node, second element is actual name.
    max_depth: int. the max depth of the code.

    Return:
    list. updated node name list.
    """
    while len(node_name[0]) < max_depth:
        node_name[0] += "0"
    return [int(node_name[0]),node_name[1]]


def extracting_path(c_code, max_length, max_width, hash_path, hashing_table):
    """Extracting paths for a given json code.
    Input:
    json_code: json object. The json object of a snap program to be extracted.
    max_length: int. Max length of the path to be restained.
    max_width: int. Max width of the path to be restained.
    hash_path: boolean. if true, MD5 hashed path will be returned to save space.
    hashing_table: Dict. Hashing table for path.

    Return:
    walk_paths: list of AST paths from the json code.
    """
    
    # Initialize head node of the code.
    head = Node(["1",get_token(c_code)])
    
    # Recursively construct AST tree.
    
    for child_order in range(len(get_children(c_code))):

        get_trees(get_children(c_code)[child_order], head, "1"+str(int(child_order)+1))
    
    # Getting leaf nodes.
    leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)
    
    # Getting max depth.
    max_depth = max([len(node.name[0]) for node in leaf_nodes])
    
    # Node rank modification.
    for leaf in leaf_nodes:
        leaf.name = get_node_rank(leaf.name,max_depth)
    
    walker = Walker()
    text_paths = []
    
    # Walk from leaf to target
    for leaf_index in range(len(leaf_nodes)-1):
        for target_index in range(leaf_index+1, len(leaf_nodes)):
            raw_path = walker.walk(leaf_nodes[leaf_index], leaf_nodes[target_index])
            
            # Combining up and down streams
            walk_path = [n.name[1] for n in list(raw_path[0])]+[raw_path[1].name[1]]+[n.name[1] for n in list(raw_path[2])]
            text_path = "@".join(walk_path)
            
            # Only keeping satisfying paths.
            if get_path_length(walk_path) <= max_length and get_path_width(raw_path) <= max_width:
                if not hash_path:
                # If not hash path, then output original text path.
                    text_paths.append(walk_path[0]+","+text_path+","+walk_path[-1])
                else:
                # If hash, then output hashed path.
                    text_paths.append(walk_path[0]+","+hashing_path(text_path, hashing_table)+","+walk_path[-1])
            
            if len(text_paths) > 200: return text_paths
    
    return text_paths


def file2path(filename):
    number = filename.split('\\')[-1].split('_')[1][1:]
    print(filename)

    txt, ast = translate_to_c(filename)
    AST_paths = []
    hashing_table = {}
    AST_paths = extracting_path(ast, max_length=8, max_width=2, hash_path=True, hashing_table=hashing_table)

    return number, AST_paths

from tqdm import tqdm

if __name__ == "__main__":
    # filename = 'EOJDataset/contest513/ContestCode\\10215102508_陈鑫_王老师_102092\\1106_#3026056_Accepted.c'
    # number, paths = file2path(filename)
    # print(paths)
    # print(number)

    code_dir = 'EOJDataset/contest513/ContestCode'
    code_files = []
    for root, dirs, files in os.walk(code_dir):
        for file in files:
            if 'Compilation' not in file:
                code_files.append(os.path.join(root, file))
    
    paths = {}
    print(len(code_files))
    cnt = 0
    for file in tqdm(code_files,mininterval=3):
        number, path = file2path(file)
        paths[number] = "@".join(path)
        cnt += 1
        if cnt % 5000 == 0:
            main_df = pd.DataFrame.from_dict(paths, orient='index')
            main_df.to_csv("EOJDataset/contest513/labeled_"+str(cnt)+"_paths.tsv", sep="\t", header=True)
    
    main_df = pd.DataFrame.from_dict(paths, orient='index')
    main_df.to_csv("EOJDataset/contest513/labeled_paths.tsv", sep="\t", header=True)



    # ast.show()
    # print(ast)
    # file = '''
    # int main(){
    #     int a,b,c;
    #     char d[] = "asbssa";
    #     scanf("%d%d",&a,&b);
    #     c = a + b;
    #     printf("%d",c);
    #     return 0;
    # }   
    
    # '''
    # txt, ast = translate_to_c_txt(file)
    # t = ast
    # for code in t:
    #     t = code
    # for code in t:
    #     t = code
    # for code in t:
    #     t = code
    # for code in t:
    #     t = code
    # t.show()
    # print(len(t.children())==0)

    
   
