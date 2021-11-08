import os
import random
import torchaudio
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from torchaudio.sox_effects import apply_effects_file
from collections import defaultdict

SAMPLE_RATE = 16000
LONG_ENOUGH_SECS = 4
TRIM_START_SECS = 2

class LxtSid(Dataset):
    def __init__(self, split, lxt_audio, utts, min_secs=2, max_secs=4, seed=0, n_train=10, n_eval=40, dump_dir=None, **kwargs) -> None:
        super().__init__()
        random.seed(seed)

        self.lxt_audio = Path(lxt_audio)
        with open(utts) as file:
            def process_line(uid):
                path = self.lxt_audio / f"{uid.strip()}.wav"
                wav, sr = torchaudio.load(str(path))
                wav = wav.squeeze(0)
                assert sr == SAMPLE_RATE

                if min_secs < 0 or max_secs < 0:
                    yield wav, 0, -1
                else:
                    if (len(wav) / SAMPLE_RATE) > LONG_ENOUGH_SECS:
                        # prevent microphone noises
                        wav = wav[TRIM_START_SECS * SAMPLE_RATE:]
                    frames = len(wav)
                    start = 0
                    while (frames - start) / SAMPLE_RATE > min_secs:
                        interval = random.randint(min_secs * SAMPLE_RATE, max_secs * SAMPLE_RATE)
                        end = start + interval
                        yield wav[start:end], start, end
                        start = end

            spk2utt = defaultdict(lambda: defaultdict(list))
            seg2wavs = {}
            for line in tqdm(file.readlines()):
                uid, spk = line.strip().split(maxsplit=1)
                for seg_wav, start, end in list(process_line(uid)):
                    seg_id = f"{uid}_{start}_{end}"
                    spk2utt[spk][uid].append(seg_id)
                    seg2wavs[seg_id] = seg_wav

            segs_with_spk = []
            for spk, utt2segs in spk2utt.items():
                utts = utt2segs.keys()
                random.shuffle(utts)

                interval = round(len(utts) / 3)
                train_utts = utts[ : interval]
                dev_utts = utts[interval : 2 * interval]
                test_utts = utts[2 * interval : ]

                all_train_segs = []
                for utt in train_utts:
                    all_train_segs.extend(utt2segs[utt])
                train_segs = random.sample(all_train_segs, k=n_train)

                all_dev_segs = []
                for utt in dev_utts:
                    all_dev_segs.extend(utt2segs[utt])
                dev_segs = random.sample(all_dev_segs, k=n_eval // 2)

                all_test_segs = []
                for utt in test_utts:
                    all_test_segs.extend(utt2segs[utt])
                test_segs = random.sample(all_test_segs, k=n_eval // 2)

                if split == "train":
                    segs_with_spk.extend([(seg_id, spk) for seg_id in train_segs])
                elif split == "dev":
                    segs_with_spk.extend([(seg_id, spk) for seg_id in dev_segs])
                elif split == "test":
                    segs_with_spk.extend([(seg_id, spk) for seg_id in test_segs])
                else:
                    raise ValueError

            self.pairs = []
            for seg, spk in segs_with_spk:
                self.pairs.append([seg2wavs[seg], spk, seg])

        self.spkrs = sorted(list(spk2utt.keys()))
        if dump_dir is not None:
            tgt_dir = Path(dump_dir) / split
            os.makedirs(tgt_dir, exist_ok=True)
            for wav, spk, seg_id in self.pairs:
                torchaudio.save(str(tgt_dir / f"{spk.replace(' ', '_')}:{seg_id}.wav"), wav.view(1, -1), SAMPLE_RATE)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        wav, spkr, uids = self.pairs[index]
        label = self.spkrs.index(spkr)
        return wav.numpy(), label, uids

    @property
    def speaker_num(self):
        return len(self.spkrs)

    @staticmethod
    def collate_fn(items):
        return zip(*items)
