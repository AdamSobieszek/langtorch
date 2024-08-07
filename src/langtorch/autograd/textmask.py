import numpy as np

class TextMask:
    def __init__(self, content=None):
        if isinstance(content, TextMask):
            self.content = content.content[:]
        elif content is None:
            self.content = []
        else:
            if isinstance(content, (int, tuple)):
                content = [content]
            self.content = sorted(set((entry,) if isinstance(entry, int) else tuple(entry) for entry in content))

    def __repr__(self):
        return f"TextMask({self.content})"
    
    def __iter__(self):
        return iter(self.content)
    
    def __getitem__(self, index):
        return TextMask(np.array(self.content, dtype=object)[index])

    def __setitem__(self, key, value):

        self.content[key] = value.content if isinstance(value, TextMask) else value

    def indexify(self, idx):
        return idx if isinstance(idx, tuple) else (idx,)

    def set(self, key, text):
        from langtorch import Text
        print("1.", self.content)
        if not isinstance(text, Text):
            text = Text(text, parse=False)
        key = self.indexify(key)
        if key in [entry[:len(key)] for entry in self.content]:
            self.content = TextMask([entry for entry in self.content if entry[:len(key)] != key] + (text._full_grad_mask().add_key(key).content if len(text._full_grad_mask().content)>1 else [key])).content
        else:
            print("!!", key)
            self.content = TextMask(self.content + text.grad_mask.add_key(key).content).content
        print("2.", self.content)

    def __add__(self, other):
        if not isinstance(other, TextMask):
            try:
                other = TextMask(other)
            except:
                raise TypeError("Addition is only supported between TextMask objects")

        return TextMask(self.content + other.content)

    def __sub__(self, other):
        if not isinstance(other, TextMask):
            try:
                other = TextMask(other)
            except:
                raise TypeError("Subtraction is only supported between TextMask objects")

        new_entries = []
        for entry in self.content:
            if not any(self._is_prefix(other_entry, entry) for other_entry in other.content):
                new_entries.append(entry)

        return TextMask(new_entries)

    @staticmethod
    def _is_prefix(prefix, entry):
        if isinstance(prefix, int):
            prefix = (prefix,)
        return entry[:len(prefix)] == prefix

    def starts_with(self, key, strip=False):
        key = self.indexify(key)
        result = self[[TextMask._is_prefix(key, entry) for entry in self.content]]
        if strip:
            result.content = [(0,)+i[len(key):] for i in result.content]
        return result

    def __eq__(self, other):
        if isinstance(other, (bool)):
            return False
        other = self.indexify(other)
        return np.array([entry == other for entry in self.content])

    def __lt__(self, other):
        other = self.indexify(other)
        return np.array([entry[:len(other)] < other for entry in self.content])
    def __gt__(self, other):
        other = self.indexify(other)
        return np.array([entry[:len(other)] > other for entry in self.content])
    def __le__(self, other):
        other = self.indexify(other)
        return np.array([entry[:len(other)] <= other for entry in self.content])
    def __ge__(self, other):
        other = self.indexify(other)
        return np.array([entry[:len(other)] >= other for entry in self.content])

    def __hash__(self):
        return hash(tuple(self.items))

    def add_key(self, key=(0,)):
        return TextMask([key+(entry) for entry in self.content])

    def shift_by(self, shift):
        return TextMask([(shift + entry[0],)+entry[1:] for entry in self.content])

    def delete(self, deleted_idx):
        def adjust_indices(idx, deleted_idx):
            if isinstance(deleted_idx, tuple) and (isinstance(idx, int) or len(idx) < len(deleted_idx)):
                return idx
            if isinstance(idx, int):
                return idx - 1 if isinstance(deleted_idx, int) and idx > deleted_idx else idx
            else:
                if isinstance(deleted_idx, int):
                    if idx[0] > deleted_idx:
                        return (idx[0] - 1,) + idx[1:]
                    else:
                        return idx
                new_idx = list(idx)
                for i in range(min(len(new_idx), len(deleted_idx))):
                    if new_idx[i] == deleted_idx[i]:
                        if i == len(deleted_idx) - 1:
                            new_idx[i] = adjust_indices(new_idx[i], deleted_idx[-1])
                    elif new_idx[i] > deleted_idx[i]:
                        new_idx[i] -= 1
                        break
                    else:
                        break
                return tuple(new_idx)
        deleted_idx = self.indexify(deleted_idx)
        return TextMask([adjust_indices(entry, deleted_idx) for entry in self.content if entry[:len(deleted_idx)] != deleted_idx])

    def delete_(self, idx):
        self.content = self.delete(idx).content
        return self

    def copy(self):
        return TextMask(self.content[:])

    def contiguous_portions(self, other=None):
        if not isinstance(other, TextMask):
            raise TypeError("Comparison is only supported between TextMask objects")

        print(self.content,other.content)
        if not set(self.content).issubset(set(other.content)):
            raise ValueError("The current TextMask must be a subset of the full TextMask. There is likely a bug in langtorch code")

        result = []
        if not self.content:
            return result
        start_index = min(self.content)
        for i in other[other>=start_index]:
            mask1_indices = self[self>=start_index]
            mask2_indices = other[(other>=start_index) & (other <= i)]

            if set(mask2_indices) - set(mask1_indices):
                result.append((start_index, last_index))
                start_index = i

            last_index = i

        # Check if the last portion is contiguous
        if not result:
            result.append((start_index, i))

        return result