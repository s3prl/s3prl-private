import torch
import argparse
import torchaudio
from tqdm import tqdm
from pathlib import Path
from librosa.util import find_files
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(args.device)
model.eval()

audios = find_files(args.audio)
with open(args.output, "w") as file:
    for audio in tqdm(audios):
        wav, sr = torchaudio.load(audio)
        wav = wav.view(-1)
        # tokenize
        input_values = processor([wav.numpy()], return_tensors="pt", padding="longest", sampling_rate=16000).input_values  # Batch size 1
        input_values = input_values.to(args.device)

        with torch.no_grad():
            # retrieve logits
            logits = model(input_values).logits.detach().cpu()

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)
        print(Path(audio).stem, transcription[0], file=file)
