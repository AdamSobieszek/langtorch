from typing import List, Union, Tuple, Any

from markdown_it import MarkdownIt

from .text import Text
from ..grammars import utils


class Node:
    def __init__(self, node_type: str, content: Any = None):
        self.node_type = node_type
        self.children = [] if content is None else [content]


class Markdown(Text):
    language = 'md'
    block_to_markdown = utils.block_to_markdown

    @classmethod
    def parse(cls, arg: str) -> List[Union[str, Tuple[str, Union[str, List]]]]:
        md = MarkdownIt()
        tokens = md.parse(arg)

        root = Node('root')
        stack = [root]
        key_mapping = {'ul': 'BulletList', 'ol': 'OrderedList', 'h1': 'Header1', 'h2': 'Header2',
                       'h3': 'Header3', 'h4': 'Header4', 'h5': 'Header5', 'h6': 'Header6'}

        for token in tokens:
            if token.nesting == 1:  # Opening tags
                new_node = Node(key_mapping.get(token.tag, token.tag))
                stack[-1].children.append(new_node)
                stack.append(new_node)
            elif token.nesting == -1:  # Closing tags
                stack.pop()
            elif token.type == 'inline':
                stack[-1].children.append(token.content)

        def tree_to_tuples(node: Node) -> Any:
            if not node.children:
                return None
            if all(isinstance(child, str) for child in node.children):
                return (node.node_type, ' '.join(node.children))
            simplified_children = [(tree_to_tuples(child) if isinstance(child, Node) else child) for child in
                                   node.children]
            # Simplify cases where leaf nodes are a list of one paragraph to just the content.
            if len(simplified_children) == 1 and isinstance(simplified_children[0], tuple) and simplified_children[0][
                0] in ['p']:
                return (node.node_type, simplified_children[0][1])
            return (node.node_type, simplified_children)

        return [tree_to_tuples(child) for child in root.children]
