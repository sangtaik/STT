import librosa
import numpy as np
import torch
from unicode import join_jamos
import re
import sys

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

class ServiceModel():
    def __init__(self,base='jamo'):
        self.base = base
        if base=='jamo':
            vocab = 'vocab_jamos.json'
            model = 'jamo_base_model'
        elif base=='char':
            vocab = 'vocab.json'
            model = 'char_base_model'
        else:
            sys.exit("'jamo' or 'char'")
            
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )        
        self.tokenizer = Wav2Vec2CTCTokenizer(
            "./Assets/"+vocab,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|"
        )
        
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer
        )
        self.model = Wav2Vec2ForCTC.from_pretrained(
            "./Assets/"+model, 
            attention_dropout=0.1,
            hidden_dropout=0.1,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.1,
            ctc_loss_reduction="mean", 
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer)
        )

    # def voice_sep(sig):
    #     sig = np.array(sig).flatten()
    #     S_full, phase = librosa.magphase(librosa.stft(sig))
    #     S_filter = librosa.decompose.nn_filter(S_full,
    #                                     aggregate=np.median,
    #                                     metric='cosine',
    #                                     width=int(librosa.time_to_frames(2, sr=sr)))
    #     S_filter = np.minimum(S_full, S_filter)
    #     margin_v = 2
    #     power = 2
    #     mask_v = librosa.util.softmask(S_full - S_filter,
    #                             margin_v * S_filter,
    #                             power=power)
    #     S_foreground = mask_v * S_full
    #     y_foreground = librosa.istft(S_foreground * phase)
    #     return y_foreground

    
    def prepare_dataset(self, batch):
        
        batch["input_values"] = self.processor(batch["array"], sampling_rate=16000).input_values[0]
        
        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        return batch

    def model_forward(self, array):
        array = self.processor(array, sampling_rate=16000).input_values[0]
        pred = self.model.forward(torch.from_numpy(array.reshape(1,-1)))
        return pred

    def pred_decode(self, pred):
        pred_logits = pred['logits'].detach().numpy()
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = self.processor.batch_decode(pred_ids)
        return pred_str
    
    def char_one_shot(self, array):
        pred = self.model_forward(array)
        pred_str = self.pred_decode(pred)
        return pred_str[0]
    
    def jamo_one_shot(self, array):
        pred = self.model_forward(array)
        pred_str = self.pred_decode(pred)
        remove_pad_token = re.sub('<pad>','',pred_str[0])
        join_jamo = join_jamos(remove_pad_token)
        return join_jamo
    
    def one_shot(self, array):
        base = self.base
        if base=='jamo':
            return self.jamo_one_shot(array)
        elif base=='char':
            return self.char_one_shot(array)
        

if __name__ == '__main__':

    service1 = ServiceModel(base='char')
    service2 = ServiceModel(base='jamo')
    array,_ = librosa.load('./Assets/test_data.wav',16000)
    print(type(array))
    print(array.shape)
    print(array.dtype)
    
    print('char_base_model :',service1.one_shot(array))
    print()
    print('jamo_base_model :',service2.one_shot(array))
    print()