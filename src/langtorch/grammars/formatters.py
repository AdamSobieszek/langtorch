import numpy as np
from itertools import zip_longest

from .text_content_ast import is_terminal_node
from .utils import block_to_markdown
from ..session import ctx
import torch


def f_markdown(instance: 'Text') -> str:
    # TODO Emph and Bold
    return "\n\n".join([block_to_markdown(instance.__class__, (k, v)) for (k, v) in instance.items()])




def f_xml(instance: 'Text') -> str:
    f = lambda k, v: f(k, "".join(f(kk, vv) for kk, vv in v)) if isinstance(v, list) else (
        f"<{k}>{f('', v)}</{k}>" if k else (v if isinstance(v, str) else f("", v)))
    return "".join([f(k, v) for (k, v) in instance.items()])


def f_concatenate_terminals(instance: 'Text') -> str:
    if hasattr(instance, "content"):
        instance = instance.items()
    result = ""

    # Base cases
    if is_terminal_node(instance):
        return instance if isinstance(instance, str) else instance[1]

    # Recursive cases
    if isinstance(instance, tuple) and len(instance) == 2 and isinstance(instance[0], str) and isinstance(instance[1], (
    tuple, list)):
        return ''.join((f_concatenate_terminals(child) for child in instance[1]) if isinstance(instance[1],
                                                                                               list) else f_concatenate_terminals(
            instance[1]))
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
import numpy as np
from itertools import zip_longest


def wrap_line(line, max_width):
    if len(line) <= max_width:
        return [line]
    return [line[i:i + max_width - 1] + '↵' if i + max_width < len(line) else line[i:i + max_width - 1] for i in
            range(0, len(line), max((1,max_width - 1)))]


def truncate_line(line, max_width):
    if len(line) > max_width:
        return line[:max_width - 3] + '...'
    return line


def process_line(line, max_width, wrap_text):
    if wrap_text:
        return wrap_line(line, max_width)
    return [truncate_line(line, max_width)]


def format_entry(entry, max_lines, max_width, wrap_text):
    # Split the entry into lines, process them (truncate or wrap), pad them to match the max width,
    # and ensure each entry has the same number of lines
    lines = str(entry).split('\n')
    processed_lines = [l for line in lines for l in process_line(str(line), max_width, wrap_text)]

    padded_lines = [str(line).ljust(max_width) for line in processed_lines]
    padded_lines += [' ' * max_width] * (max_lines - len(processed_lines))  # Pad with empty lines if needed
    return padded_lines


def calculate_optimal_column_widths(array_2d, max_width):
    n_columns = len(array_2d[0])

    # Calculate initial max widths for each column
    max_width_per_col = np.array([
        min(max(max([len(str(line)) for line in str(element).split('\n')]) for element in col), max_width // n_columns)
        for col in zip(*array_2d)])
    width_per_col = np.array([
        max(max([len(str(line)) for line in str(element).split('\n')]) for element in col)
        for col in zip(*array_2d)])

    total_width = sum(max_width_per_col)

    while total_width < max_width and any(width_per_col>max_width_per_col):
        for i in range(n_columns):
            if width_per_col[i] > max_width_per_col[i]:
                max_width_per_col[i] += 1
                total_width += 1
                if total_width >= max_width:
                    break
    max_width_per_col[max_width_per_col>(max_width // n_columns)*3] = (max_width // n_columns)*3

    return max_width_per_col

def format_2d(array_2d, max_width=200, wrap_text=False, indent=" ", spacing=2):
    # Calculate max width for each column, but limit the width to column_limit
    if len(indent) == 0: indent = ' '
    max_width_per_col = calculate_optimal_column_widths(array_2d, max_width)
    # Create a list of lists for each formatted line
    formatted_rows = []
    spacer = ' ' * spacing
    for row in array_2d:
        max_lines_in_row = max(
            len(process_line(str(line).replace("\n", ""), max_width_per_col[i], wrap_text)) for i, line in
            enumerate(row))
        formatted_entries = [format_entry(entry, max_lines_in_row, max_width, wrap_text)
                             for entry, max_width in zip(row, max_width_per_col)]
        max_lines_in_row = max(len(m) for m in formatted_entries)

        formatted_entries = [m + [' ' * len(m[0])] * (max_lines_in_row - len(m)) for i, m in
                             enumerate(formatted_entries)]

        transposed_entries = list(zip_longest(*formatted_entries, fillvalue=' ' * max(max_width_per_col)))

        for j, transposed_line in enumerate(transposed_entries):
            line = (spacer).join(transposed_line)
            if j == 0:
                line = indent[1:] + '[' + line + (']' if len(transposed_entries) == 1 else ' ')
            elif j == len(transposed_entries) - 1:
                line = indent + line + ']'
            else:
                line = indent + line + ' '
            formatted_rows.append(line)
    return formatted_rows


def format_1d_as_row(array_1d, column_limit=200, wrap_text=False):
    formatted_rows = format_2d(array_1d.reshape(1, -1), column_limit, wrap_text, spacing=3)
    return ('\n' + " ").join(formatted_rows)


def format_1d(array_1d, column_limit=200, wrap_text=False):
    # Treat the 1D array as a 2D array with a single column
    return [m[1:-1] if len(m) >= 2 else m for m in format_2d(array_1d.reshape(-1, 1), column_limit, wrap_text)]


def tensor_str_formatter(cls, array: np.ndarray, indent: str = " ") -> str:
    max_width = ctx.max_width
    wrap_text = ctx.soft_wrap
    if array.ndim == 0:
        return str(array.item())
    if array.ndim == 1:
        row_repr = format_1d_as_row(array, max_width, wrap_text)
        if len(row_repr.split("\n")) > 4 * max([len(str(t).split('\n')) for t in array.flat]):
            formatted_lines = format_1d(array, max_width, wrap_text)
            return '[' + ('\n' + indent).join(formatted_lines) + ']'
        else:
            return format_1d_as_row(array, max_width, wrap_text)
    elif array.ndim == 2:
        indent = ' '
        formatted_lines = format_2d(array, max_width, wrap_text, spacing=3)
        return indent[1:] + '[' + ('\n' + indent).join(formatted_lines) + ']'
    else:
        inner_arrays = [indent + tensor_str_formatter(cls, sub_array, indent + '  ') for sub_array in array]
        inner_content = (',\n').join(("\n" + indent).join(a.split("\n")) for a in inner_arrays)
        return '[\n' + inner_content + '\n' + indent[:-2] + ']'
