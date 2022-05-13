import pyarrow as pa
from tqdm import tqdm

import torch
import torch.nn as nn

from s3prl import hub
from s3prl.base.output import Output
from s3prl.util.gpu_dataloader import MultipleGPUDataLoader
from s3prl.util.benchmark import benchmark

BATCH_SIZE = 8


class Project(nn.Module):
    def __init__(self):
        super().__init__()
        self.upstream = getattr(hub, "wav2vec2_large_ll60k")()

    def forward(self, wav):
        hidden_states = self.upstream(wav)["hidden_states"]
        hidden_states = torch.stack(hidden_states, dim=2)
        return hidden_states


class simple_iterator:
    def __init__(self) -> None:
        self.num = 1000

    def __len__(self):
        return self.num

    def __iter__(self):
        for i in range(self.num):
            yield Output(wav=[torch.randn(16000 * 10) for i in range(8)])


def test_multigpu_dataloader():
    dataloader = MultipleGPUDataLoader(simple_iterator(), Project, (), 1, [2, 0])
    for batch_id, batch in enumerate(tqdm(dataloader)):
        pass


def test_singlegpu_dataloader():
    device_id = 1
    model = Project().cuda(device_id)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(simple_iterator()):
            batch.to(f"cuda:{device_id}")
            model(**batch)


def test_speed():
    model = Project().cuda(0)
    model.eval()
    with torch.no_grad():
        for batch in simple_iterator():

            with benchmark("move wavs", device=f"cuda:0", freq=1):
                batch = batch.to(f"cuda:0")

            with benchmark("forward", device=f"cuda:0", freq=1):
                result = model(**batch)

            with benchmark("move to cpu", device="cuda:0", freq=1):
                result: torch.Tensor = result.cpu()
                result = result.share_memory_()

            with benchmark("move to cuda", device="cuda:0", freq=1):
                result = result.to(f"cuda:1")

            print(
                pa.serialize(result.detach().cpu().numpy()).to_buffer().size
                / (1024**3),
                "GB",
            )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    test_multigpu_dataloader()
    # test_singlegpu_dataloader()
