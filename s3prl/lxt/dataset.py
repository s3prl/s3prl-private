import pickle
from typing import List
from pathlib import Path

import torchaudio


class UttrInfo:
    def __init__(self, audio_path) -> None:
        self.id: str = Path(audio_path).stem
        info = torchaudio.info(str(audio_path))
        seconds = info.num_frames / info.sample_rate
        self.seconds: float = seconds


class UttrCollection:
    @property
    def all_uttrs(self):
        return len(self.uttrs)

    @property
    def all_seconds(self):
        return sum([uttr.seconds for uttr in self.uttrs])


class SpkrInfo(UttrCollection):
    def __init__(self, spkr_name, csv, audio_dir) -> None:
        self.uttrs: List[UttrInfo] = []
        self.gender = 0 if "female" in spkr_name else 1
        self.name = spkr_name
        for _, row in csv[csv["speaker (gender/id)"] == self.name].iterrows():
            audio_path = Path(audio_dir) / f"{row['utterance_id']}.wav"
            if not audio_path.is_file():
                print(audio_path, "not found.")
                continue
            self.add_uttr(audio_path)
    
    def __str__(self) -> str:
        return self.name

    def add_uttr(self, uid):
        self.uttrs.append(UttrInfo(uid))


class SpkrCollection:
    @property
    def spkr_num(self):
        return len(self.spkrs)

    @property
    def male(self):
        return [spkr for spkr in self.spkrs if spkr.gender == 1]

    @property
    def male_num(self):
        return len(self.male)

    @property
    def female(self):
        return [spkr for spkr in self.spkrs if spkr.gender == 0]

    @property
    def female_num(self):
        return len(self.female)

    @property
    def male_uttrs(self):
        return sum([spkr.all_uttrs for spkr in self.spkrs if spkr.gender == 1])

    @property
    def female_uttrs(self):
        return sum([spkr.all_uttrs for spkr in self.spkrs if spkr.gender == 0])

    @property
    def male_seconds(self):
        return sum([spkr.all_seconds for spkr in self.spkrs if spkr.gender == 1])

    @property
    def female_seconds(self):
        return sum([spkr.all_seconds for spkr in self.spkrs if spkr.gender == 0])


class DatasetInfo(UttrCollection, SpkrCollection):
    def __init__(self, spkrs: List[SpkrInfo], name: str) -> None:
        self.spkrs: List[SpkrInfo] = spkrs
        self.name = name

    @property
    def uttrs(self):
        uttrs: List[UttrInfo] = []
        for spkr in self.spkrs:
            uttrs.extend(spkr.uttrs)
        return uttrs

    @property
    def uttr_ids(self):
        return [uttr.id for uttr in self.uttrs]

    @property
    def male_dataset(self):
        return DatasetInfo(self.male, name=f"{self.name}_male")

    @property
    def female_dataset(self):
        return DatasetInfo(self.female, name=f"{self.name}_female")

    def __str__(self) -> str:
        message = ""
        message += f"uttrs: {self.all_uttrs}\n"
        message += f"seconds: {self.all_seconds}\n"
        message += f"hours: {self.all_seconds / 60 / 60}\n"
        message += f"spkrs: {self.spkr_num}\n"
        message += f"males: {self.male_num}\n"
        message += f"females: {self.female_num}\n"
        message += f"male uttrs: {self.male_uttrs}\n"
        message += f"female uttrs: {self.female_uttrs}\n"
        message += f"male seconds: {self.male_seconds}\n"
        message += f"male hours: {self.male_seconds / 60 / 60}\n"
        message += f"female seconds: {self.female_seconds}\n"
        message += f"female hours: {self.female_seconds / 60 / 60}\n"
        return message

    def save(self, filepath):
        with open(str(filepath), "wb") as pkl:
            pickle.dump(self, pkl)

