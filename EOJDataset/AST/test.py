from tree_sitter import Language, Parser

# Language.build_library(
#   # Store the library in the `build` directory
#   'build/my-languages.so',

#   # Include one or more languages
#   [
#     'tree-sitter-c',
#   ]
# )
C_LANGUAGE = Language('EOJDataset/AST/build/my-languages.so', 'c')

parser = Parser()
parser.set_language(C_LANGUAGE)

tree = parser.parse(bytes("""
def foo():
    if bar:
        baz()
""", "utf8"))