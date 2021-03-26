import datasets
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import torch
from jiwer import wer
import librosa
import re
import json


chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

modelName = "flozi00/wav2vec-xlsr-german"
language = "de"


model = Wav2Vec2ForCTC.from_pretrained(modelName).to("cuda")
tokenizer = Wav2Vec2Processor.from_pretrained(modelName)


print("loading and cleaning")
val_dataset = datasets.load_dataset("common_voice", language, split="train+test+validation")
val_dataset = val_dataset.remove_columns(["client_id","up_votes","down_votes","age","gender","accent","locale","segment"])
val_dataset = val_dataset.rename_column("path","file")
val_dataset = val_dataset.rename_column("sentence","text")




def map_to_array(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper() + " "
    try:
        speech_array, sampling_rate = sf.read(batch["file"]+ ".wav")
    except:
        speech_array, sampling_rate = librosa.load(batch["file"], sr=16000, res_type='kaiser_fast')
        sf.write(batch["file"] + ".wav", speech_array, sampling_rate, subtype='PCM_24')
    batch["speech"] = speech_array
    return batch


print("reading audio")
val_dataset = val_dataset.map(map_to_array)


def map_to_pred(batch):
    inputs = tokenizer(batch["speech"], return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    with open('t5-dataset.json', 'a+') as outfile:
        for el in range(len(transcription)):
            if(wer(transcription[el], batch['text'][el]) < 0.4):
                entry = {"translation": {"transcribed": transcription[el], "corrected": batch['text'][el]}}
                json.dump(entry, outfile)
                outfile.write('\n')

    batch["transcription"] = transcription
    return batch


print("predicting")
result = val_dataset.map(map_to_pred, batched=True, batch_size=32, remove_columns=["speech"])

print("WER:", wer(result["text"], result["transcription"]))