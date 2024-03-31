import os
from functools import wraps


def override(func):
    @wraps(func)
    def wrapper(*args, save_filepath=None, override=False, **kwargs):
        if override and save_filepath:
            abs_filepath = os.path.abspath(save_filepath)
            dir_path = os.path.dirname(abs_filepath)
            base_filename = os.path.splitext(os.path.basename(abs_filepath))[0]
            log_file_path = os.path.join(dir_path, base_filename + "_log.txt")
            ordered_file_path = os.path.join(dir_path, base_filename + "_log.txt")

            # Empty the contents of the save_filepath and the file_path.
            with open(abs_filepath, 'w'):
                pass

            with open(log_file_path, 'w'):
                pass

            if os.path.exists(ordered_file_path):
                with open(ordered_file_path, 'w'):
                    pass

        # Check if 'save_filepath' is in kwargs
        if 'save_filepath' in kwargs:
            kwargs['save_filepath'] = save_filepath

        return func(*args, **kwargs)

    return wrapper
