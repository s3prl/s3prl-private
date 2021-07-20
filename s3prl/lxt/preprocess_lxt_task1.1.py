import os
import torch
import shutil
import argparse
import torchaudio
from tqdm import tqdm
from shutil import copy
from pathlib import Path
from collections import defaultdict

LXT_SAMPLE_RATE = 44100

parser = argparse.ArgumentParser()
parser.add_argument("--lxt", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--copy", action="store_true")
args = parser.parse_args()

delete = [
    "quac-C_8f1052a909924795a456beaa053ffe22_0_q#6-q-0.wav",  # our script is wrong
    "newsqa-1f131129a9a54e9f8fa214e76b6e9b9f-c-3.wav",  # no recording
    "quac-C_6c922ff8e4a340d9bf2b85558179fdde_0_q#4-c-3.wav",  # no recording
    "quac-C_7d5560b23977415f9b2fda38a973e3b6_0_q#10-c-13.wav",  # no recording
    "squad-5ad24eb1d7d075001a428c62-c-5.wav",  # wrong recording
    "quac-C_898087a45105402ab04da342233b1341_1_q#6-c-4.wav",  # wrong recording
    "quac-C_4eb70ac453504815b1a0422105098435_0_q#2-c-3.wav",  # wrong recording
    "quac-C_5d024d0f9d5e431aa844c767ab6d26df_1_q#6-c-1.mp3",  # mp3 file
    "squad-5ad3a266604f3c001a3fea29-c-6(1).mp3",  # mp3 file
    "squad-5ad3a266604f3c001a3fea29-c-6.mp3",  # mp3 file
    "squad-5ad3a266604f3c001a3fea29-q-0(1).wav",  # duplicated
    "squad-5ad3f5b0604f3c001a3ff9ac-c-0(1).wav",  # duplicated
    "squad-5ad3f5b0604f3c001a3ff9ac-c-4(1).wav",  # duplicated
    "squad-5ad40219604f3c001a3ffd47-q-0(1).wav",  # duplicated
    "squad-5ad4e4425b96ef001a10a573-c-1(1).wav",  # duplicated
    "squad-5ad55ee35b96ef001a10ace8-c-4(1).wav",  # duplicated
]

rename = [
    ("quac-C_8f1052a909924795a456beaa053ffe22_0_q#6-c-12.wav", "quac-C_8f1052a909924795a456beaa053ffe22_0_q#6-c-11.wav"),
    ("quac-C_8f1052a909924795a456beaa053ffe22_0_q#6-c-11.wav", "quac-C_8f1052a909924795a456beaa053ffe22_0_q#6-c-12.wav"),
    ("quac-C_63779f1d353d42d797a91b1c93ad49a3_0_q#6-c-11.wav", "quac-C_63779f1d353d42d797a91b1c93ad49a3_0_q#6-c-0.wav"),
    ("quac-C_63779f1d353d42d797a91b1c93ad49a3_0_q#6-c-11(1).wav", "quac-C_63779f1d353d42d797a91b1c93ad49a3_0_q#6-c-11.wav"),
    ("newsqa-b020254fcd674482ade2f033be2149db-c-2.wav", "newsqa-b020254fcd674482ade2f033be2149db-c-7.wav"),
    ("newsqa-b020254fcd674482ade2f033be2149db-c-7.wav", "newsqa-b020254fcd674482ade2f033be2149db-c-2.wav"),
    ("quac-C_8812f60078f14c69a724af179e567d72_1_q#8-q-0 (2).wav", "quac-C_8812f60078f14c69a724af179e567d72_1_q#8-q-0.wav"),
    ("quac-C_ae03fafc4b924bbf895654156aa0f02b_1_q#2-c-11 (2).wav", "quac-C_ae03fafc4b924bbf895654156aa0f02b_1_q#2-c-11.wav"),
    ("newsqa-3c87c1c25d1e4911baef69865cf0249d-c-8c.wav", "newsqa-3c87c1c25d1e4911baef69865cf0249d-c-8.wav"),
    ("newsqa-74e4d04951fb4936b518a4962cf1f7f2-c-0 (2).wav", "newsqa-74e4d04951fb4936b518a4962cf1f7f2-c-0.wav"),
    ("quac-C_a4a81e05163b4289a304a52b590518ae_1_q#0-c-0(1).wav", "quac-C_a4a81e05163b4289a304a52b590518ae_1_q#0-c-11.wav"),
    ("C_3e7c4d0e33504faeb7704c042414bc79_0_q#5-c-2.wav", "quac-C_3e7c4d0e33504faeb7704c042414bc79_0_q#5-c-2.wav"),
    ("quac-C_3e7c4d0e33504faeb7704c042414bc79_0_q#1-c-2_.wav", "quac-C_3e7c4d0e33504faeb7704c042414bc79_0_q#1-c-2.wav"),
    ("newsqa-540c861d659d42c48bc51e62fccc3b12-c-6 (2).wav", "newsqa-540c861d659d42c48bc51e62fccc3b12-c-6.wav"),
    ("quac-C_f3296ffccc0141288e93452f8a063aff_1_q#3-c-0 (2).wav", "quac-C_f3296ffccc0141288e93452f8a063aff_1_q#3-c-0.wav"),
    ("quac-C_eb63b375b47a460ea37035571eb89411_0_q#4-c-7 (2).wav", "quac-C_63779f1d353d42d797a91b1c93ad49a3_0_q#6-c-6.wav"),
    ("quac-C_9ddd9c44b44a41579fc75b3070872975_1_q#5-c-2 (2).wav", "quac-C_9ddd9c44b44a41579fc75b3070872975_1_q#5-c-2.wav"),
    ("newsqa-d2535331646b4acaa85a8672eb049205-c-1 (2).wav", "newsqa-d2535331646b4acaa85a8672eb049205-c-1.wav"),
    ("quac-C_b82cebc289f04e378fc6a55461067aa4_1_q#4-c-9 (2).wav", "quac-C_b82cebc289f04e378fc6a55461067aa4_1_q#4-c-9.wav"),
    ("quac-C_58167e2e608c425bbb6d42b4b26f5419_0_q#3-c-0 (2).wav", "quac-C_58167e2e608c425bbb6d42b4b26f5419_0_q#3-c-0.wav"),
]

concat = [
    ("squad-572871bc4b864d1900164a04-c-0 (1).wav", "squad-572871bc4b864d1900164a04-c-0 (2).wav", "squad-572871bc4b864d1900164a04-c-0.wav"),
    ("squad-572885c44b864d1900164a7b-c-0 (1).wav", "squad-572885c44b864d1900164a7b-c-0 (2).wav", "squad-572885c44b864d1900164a7b-c-0.wav"),
    ("squad-5a6790dbf038b7001ab0c2d1-c-3 (1).wav", "squad-5a6790dbf038b7001ab0c2d1-c-3 (2).wav", "squad-5a6790dbf038b7001ab0c2d1-c-3.wav"),
    ("quac-C_49df1a3d8a104ac18720c7e0be7a2aa3_1_q#0-c-9 (1).wav", "quac-C_49df1a3d8a104ac18720c7e0be7a2aa3_1_q#0-c-9 (2).wav", "quac-C_49df1a3d8a104ac18720c7e0be7a2aa3_1_q#0-c-9.wav")
]

src_dir = Path(args.lxt)
tgt_dir = Path(args.output)
assert src_dir.is_dir()
if tgt_dir.is_dir():
    shutil.rmtree(tgt_dir)
tgt_dir.mkdir()

src_preocessed = defaultdict(lambda: False)
for filename in delete:
    src_preocessed[filename] = True

for old_filename, new_filename in rename:
    src_preocessed[old_filename] = True
    (tgt_dir / new_filename).symlink_to((src_dir / old_filename).resolve())

for filename1, filename2, new_filename in concat:
    src_preocessed[filename1] = True
    src_preocessed[filename2] = True

    wav1, sr1 = torchaudio.load(src_dir / filename1)
    wav2, sr2 = torchaudio.load(src_dir / filename2)
    assert sr1 == sr2 == LXT_SAMPLE_RATE
    wav = torch.cat([wav1, wav2], dim=-1)
    torchaudio.save(str((tgt_dir / new_filename).resolve()), wav, LXT_SAMPLE_RATE)

for filename in src_dir.iterdir():
    if not src_preocessed[filename.name]:
        (tgt_dir / filename.name).symlink_to((src_dir / filename.name).resolve())

if args.copy:
    paths = list(tgt_dir.iterdir())
    for path in tqdm(paths, total=len(paths), desc="Copying file"):
        if path.is_symlink():
            src_path = os.readlink(str(path))
            tgt_path = str(path)
            path.unlink()
            copy(src_path, tgt_path)
