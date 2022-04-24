from typing import List
from dataclasses import dataclass


@dataclass
class CategoryEncoder:
    category: List[str]

    def __len__(self):
        return len(self.category)

    def encode(self, label):
        return self.category.index(label)

    def decode(self, index):
        return self.category[index]