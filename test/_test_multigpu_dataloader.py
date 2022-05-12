from tqdm import tqdm

import torch
import torch.nn as nn

from s3prl import hub
from s3prl.base.output import Output
from s3prl.util.gpu_dataloader import MultipleGPUDataLoader

BATCH_SIZE = 8


class Project(nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = getattr(hub, "wav2vec2_large_ll60k")()

    def forward(self, wav):
        hidden_states = self.upstream(wav)["hidden_states"]
        return Output(output=hidden_states)


class simple_iterator:
    def __init__(self) -> None:
        self.num = 1000

    def __len__(self):
        return self.num

    def __iter__(self):
        for i in range(self.num):
            yield Output(wav=[torch.randn(16000 * 10) for i in range(8)])


def test_multigpu_dataloader():
    dataloader = MultipleGPUDataLoader(simple_iterator(), Project, (), 0, [0, 1])
    for batch_id, batch in enumerate(tqdm(dataloader)):
        pass


def test_singlegpu_dataloader():
    model = Project().cuda(1)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(simple_iterator()):
            batch.to(f"cuda:1")
            model(**batch)


if __name__ == "__main__":
    test_multigpu_dataloader()
    test_singlegpu_dataloader()
