import numpy as np
from itertools import zip_longest

from .text_content_ast import is_terminal_node
from .utils import block_to_markdown


def f_markdown(instance: 'Text') -> str:
    # TODO Emph and Bold
    return "\n\n".join([block_to_markdown(instance.__class__, (k, v)) for (k, v) in instance.items()])


def f_concatenate_terminals(instance: 'Text') -> str:
    if hasattr(instance, "content"):
        instance = instance.items()
    result = ""

    # Base cases
    if is_terminal_node(instance):
        return instance if isinstance(instance, str) else instance[1]

    # Recursive cases
    if isinstance(instance, tuple) and len(instance) == 2 and isinstance(instance[0], str) and (
            isinstance(instance[1], list) or isinstance(instance[1], tuple)):
        return ''.join(f_concatenate_terminals(child) for child in instance[1])
    if isinstance(instance, list):
        return ''.join(f_concatenate_terminals(child) for child in instance)

    return result


class Formatters:
    def __getitem__(self, item):
        return {
            "str": f_concatenate_terminals,
            "md": f_markdown
        }.get(item, f_concatenate_terminals)


language_to_formatter = Formatters()


def tensor_str_formatter(cls, array: np.ndarray, indent: str = " ") -> str:
    def format_entry(entry, max_lines, max_width):
        # Split the entry into lines, pad them to match the max width, and ensure each entry has the same number of lines
        lines = entry.split('\n')
        padded_lines = [str(line).ljust(max_width) for line in lines]
        padded_lines += [' ' * max_width] * (max_lines - len(lines))  # Pad with empty lines if needed
        return padded_lines

    def format_2d(array_2d):
        # Calculate max width for each column
        max_width_per_col = [max(max([len(str(line)) for line in element.split('\n')]) for element in col) for col
                             in
                             zip(*array_2d)]

        # Create a list of lists for each formatted line
        formatted_rows = []
        for row in array_2d:
            max_lines_in_row = max(element.count('\n') + 1 for element in row)
            formatted_entries = [format_entry(entry, max_lines_in_row, max_width)
                                 for entry, max_width in zip(row, max_width_per_col)]
            transposed_entries = list(zip_longest(*formatted_entries, fillvalue=' ' * max_width_per_col[0]))

            for j, transposed_line in enumerate(transposed_entries):
                line = '  '.join(transposed_line).rstrip()
                if j == 0:
                    line = '[' + line + (']' if len(transposed_entries) == 1 else '')
                elif j == len(transposed_entries) - 1:
                    line = ' ' + line + ']'
                else:
                    line = ' ' + line + ' '
                formatted_rows.append(line)
        return formatted_rows

    def format_1d_as_row(array_1d):
        max_lines = max(element.count('\n') + 1 for element in array_1d)
        max_width = max(max(len(str(line)) for line in str(element).split('\n')) for element in array_1d)
        formatted_entries = [format_entry(entry, max_lines, max_width) for entry in array_1d]

        # Transpose to align multiline entries
        transposed_entries = list(zip_longest(*formatted_entries, fillvalue=' ' * max_width))

        formatted_rows = ['  '.join(line) for line in transposed_entries]
        return '[ ' + ('\n' + indent).join(formatted_rows) + ' ]'

    def format_1d(array_1d):
        # Treat the 1D array as a 2D array with a single columns
        return [m[1:-1] if len(m) >= 2 else m for m in format_2d(array_1d.reshape(-1, 1))]

    if array.ndim == 0:
        return str(array.item())
    if array.ndim == 1:
        if len(str(array)) > 100:
            indent = ' '
            formatted_lines = format_1d(array)
            return '[' + ('\n' + indent).join(formatted_lines) + ']'
        else:
            return format_1d_as_row(array)
    elif array.ndim == 2:
        indent = ' '
        formatted_lines = format_2d(array)
        return indent[1:] + '[' + ('\n' + indent).join(formatted_lines) + ']'
    else:
        inner_arrays = [cls.str_formatter(sub_array, indent + '  ') for sub_array in array]
        inner_content = (',\n' + indent).join(inner_arrays)
        return '[\n' + indent + inner_content + '\n' + indent[:-2] + ']'
