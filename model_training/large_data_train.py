from datasets import load_dataset
import transformers
import pandas as pd
import numpy as np
import os
import time
import torch
from tqdm import tqdm

print("large_data_train start")
print("device :","cuda" if torch.cuda.is_available() else "cpu")

## data Cleaning

import re
from unicode import split_syllables, join_jamos

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    batch["text"] = split_syllables(batch["text"])
    return batch

## tokenizer
from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab_jamos.json",
                                 unk_token="<unk>",
                                 pad_token="<pad>",
                                 word_delimiter_token="|")
## FeatureExtractor
from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                             sampling_rate=16000,
                                             padding_value=0.0,
                                             do_normalize=True,
                                             return_attention_mask=True)
## Processor
from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                              tokenizer=tokenizer)

## data augmentation
import librosa
sr = 16000
def load_audio(batch):
#     batch['array'],_ = librosa.load('./dataset/audio/'+batch['filename'],sr=16000)
    batch['array'] = np.array(batch['array'][1:-1].split(',')).astype(np.float32)
    return batch

rir_raw,_ = librosa.load('./room_component.wav',sr)
rir = torch.from_numpy(rir_raw.reshape(1,-1))
print(rir.shape)

def rir_applied(batch):

    speech = torch.from_numpy(np.array(batch['array'],dtype=np.float32).reshape(1,-1))

    speech_ = torch.nn.functional.pad(speech, (rir.shape[1] - 1, 0))
    
    augmented = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]
    batch['array'] = augmented.reshape(-1)
    return batch

def fast_stretching(batch):
    array = np.array(batch['array'],dtype=np.float32)
    batch['array'] = librosa.effects.time_stretch(array,0.8)
    return batch

def too_fast_stretching(batch):
    array = np.array(batch['array'],dtype=np.float32)
    batch['array'] = librosa.effects.time_stretch(array,0.5)
    return batch

## padding

def prepare_dataset(batch):

    # batched output is "un-batched"
    batch["input_values"] = processor(
        batch["array"],
        sampling_rate=16000
    ).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.cuda.HalfTensor]]]) -> Dict[str, torch.cuda.HalfTensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

from datasets import load_dataset, load_metric, Audio

wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data import random_split

## Lage_data_traing Start

data_path = "./dataset/csv/"
csv_list = os.listdir(data_path)
output_dir = "./wav2vec2-large-xlsr-ko-demo"

for file in tqdm(csv_list[:]):
    print()
    print("before empty_cache :",
          torch.cuda.memory_allocated(device="cuda"))
    torch.cuda.empty_cache()
    print("after empty_cache :",
          torch.cuda.memory_allocated(device="cuda"))
    
    print("data loading")
    all_data = load_dataset(
        'csv',
        data_files=os.path.join(data_path, file),
        sep='\t',
        split='train'
    )
    
    print("data remove_spectial_char")
    remove_spectial_char_data = all_data.map(remove_special_characters)
    
    print("load_audio_data")
    audio_data = remove_spectial_char_data.map(load_audio)
    
    print("rir_applied")
    rir_applied_audio_data = audio_data.map(rir_applied)
    
    print("fast_stretching")
    fast_stretching_data = rir_applied_audio_data.map(fast_stretching)
    
    print("too_fast_stretching")
    too_fast_stretching_data = rir_applied_audio_data.map(too_fast_stretching)
    
    ds_list = [
        audio_data,
        rir_applied_audio_data,
        fast_stretching_data,
        too_fast_stretching_data
    ]
    
    print("dataset prepare")
    prepare_ds_list = []
    for ds in ds_list:
        prepare_ds_list.append(ds.map(
            prepare_dataset,
            remove_columns=ds.column_names,
            # num_proc=2
        ))
    augmented_data = torch.utils.data.ConcatDataset(prepare_ds_list)
    print("concat completed")
    
    model_list = os.listdir(output_dir)
    print(model_list[0],"loading")
    
    model = Wav2Vec2ForCTC.from_pretrained(
        os.path.join(output_dir,model_list[0]), 
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean", 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    training_args = TrainingArguments(
      # output_dir="/content/gdrive/MyDrive/wav2vec2-large-xlsr-ko-demo",
      output_dir="./wav2vec2-large-xlsr-ko-demo",
      group_by_length=True,
      per_device_train_batch_size=8,
      gradient_accumulation_steps=2,
      evaluation_strategy="steps",
      num_train_epochs=20,
      fp16=True,
      save_steps=100,
      eval_steps=100,
      logging_steps=10,
      learning_rate=3e-4,
      warmup_steps=500,
      save_total_limit=2,
    #   auto_find_batch_size=True, # need -> pip install accelerate
      load_best_model_at_end=True
    )
    
    ds_size = len(augmented_data)
    train_size = int(ds_size*0.8)
    val_size = ds_size - train_size
    train_ds, val_ds = random_split(augmented_data,[train_size,val_size])

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
    )
    
    print("before empty_cache :",
          torch.cuda.memory_allocated(device="cuda"))
    torch.cuda.empty_cache()
    print("after empty_cache :",
          torch.cuda.memory_allocated(device="cuda"))
    
    trainer.train()