from typing import Union, Tuple, List
import logging

def block_to_markdown(cls, entry: Union[str, Tuple[str, Union[str, List]]]) -> str:
    newline = '\n'

    key_to_md = {
        "Header": "# ",
        'Header1': '# ',
        'Header2': '## ',
        'Header3': '### ',
        'Header4': '#### ',
        'Header5': '##### ',
        'Header6': '###### ',
        'BulletList': '- ',
        'OrderedList': '{i}. ',
        'HorizontalRule': '---',
        "Para": "",
        'blockquote': '> ',
        'code': '```',
        'p': ''
    }

    # Base cases
    if isinstance(entry, str) and not hasattr(entry, "content"):
        return entry
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str) and isinstance(entry[1], str):
        return f"{key_to_md.get(entry[0], '')}{entry[1]}"

    # Recursive cases
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str) and isinstance(entry[1],
                                                                                                 (list, tuple)):
        # Special case for code blocks
        if entry[0] == 'code':
            return f"{key_to_md.get(entry[0], '')}\n{entry[1]}\n{key_to_md.get(entry[0], '')}"
        elif entry[0] in ['BulletList', 'OrderedList', 'BlockQuote']:
            # Handling of list items
            if isinstance(entry[1], tuple):
                entry = (entry[0],[entry[1]])
            result = ""
            for i, child in enumerate(entry[1]):
                if isinstance(child, str):
                    child = ('', child)
                if result:
                    result += newline
                if entry[0] == 'OrderedList':
                    result += key_to_md.get(entry[0], '').format(i=i + 1) + block_to_markdown(cls, child[1])
                else:
                    result += key_to_md.get(entry[0], '') + block_to_markdown(cls, child[1])
            return result
        else:
            return f"{key_to_md.get(entry[0], '')}{newline.join([block_to_markdown(cls, child) for child in entry[1]])}"

    if isinstance(entry, list):
        return newline.join([block_to_markdown(cls, child) for child in entry])

    logging.warning(f"Markdown parsing error for entry: {entry}")
    return f"{entry}"
