import io
import multiprocessing
import sys

import numpy as np

from .texttensor import TextTensor
from ..texts import Code
from ..utils import zeros_like


class CodeTensor(TextTensor):
    ttype = Code

    @classmethod
    def to_array(cls, input, **kwargs):
        kwargs["parse"] = False
        return super().to_array(cls, input, **kwargs)

    @classmethod
    def input_formatter(cls, content):
        # Extract code blocks from content if they are inside triple backticks
        formatted_content = []
        for entry in content.flat:
            if "```" in entry:
                start_idx = entry.find("```") + 3
                end_idx = entry.rfind("```")
                code_entry = entry[start_idx:end_idx]
                if code_entry[:len("python")] == "python":
                    code_entry = code_entry[len("python"):]
                formatted_content.append(code_entry)
            else:
                formatted_content.append(entry)
        return np.array(formatted_content, dtype=object).reshape(content.shape)

    # TODO Code Linter

    def _execute_code(self, code_entry, namespace_entry, queue):
        # Redirect stdout to capture the output
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        namespace = dict(namespace_entry.items())
        try:
            exec(code_entry, namespace)
            queue.put(new_stdout.getvalue())
        except Exception as e:
            queue.put(str(e))
        finally:
            sys.stdout = old_stdout

    def _execute_code_serial(self, code_entry, namespace_entry):
        # Redirect stdout to capture the output
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        namespace = dict(namespace_entry.items())
        try:
            exec(code_entry, namespace)
            return new_stdout.getvalue()
        except Exception as e:
            return str(e)
        finally:
            sys.stdout = old_stdout

    def eval(self, input_text_tensor, concurrent=True):
        # Match the shape of CodeTensor's content with input_text_tensor
        namespace_tensor = zeros_like(self.content) + input_text_tensor

        if concurrent:
            # Execute code snippets concurrently using multiprocessing

            # Create a list to hold the processes and a queue for inter-process communication
            processes = []
            queue = multiprocessing.Queue()

            # Execute each code entry with its corresponding namespace
            for code_entry, namespace_entry in zip(self.content.flat, namespace_tensor.flat):
                process = multiprocessing.Process(target=self._execute_code, args=(code_entry, namespace_entry, queue))
                processes.append(process)
                process.start()

            # Collect outputs from the processes
            outputs = [queue.get() for _ in processes]

            # Ensure all processes have finished
            for process in processes:
                process.join()

        else:
            # Execute code snippets serially
            outputs = []
            for code_entry, namespace_entry in zip(self.content.flat, namespace_tensor.flat):
                output = self._execute_code_serial(code_entry, namespace_entry)
                outputs.append(output)

        # Return the outputs reshaped to the original shape
        return TextTensor(outputs).reshape(self.content.shape)
