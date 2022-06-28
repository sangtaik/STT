import onnxruntime

import json
import numpy as np
import re
from unicode import join_jamos

from symspellpy import SymSpell, Verbosity

from transformers import Wav2Vec2FeatureExtractor


class ServiceModel():
    def __init__(self):
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession("./Assets/jamo_base_model.onnx", sess_options)
        
        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )   
        
        with open('./Assets/vocab_jamos.json','r') as f:
            self.word_to_index = json.load(f)
        
        self.index_to_word = {index:word for word,index in self.word_to_index.items()}
        
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        self.sym_spell.load_dictionary("./Assets/symspell_jamo_dict.txt", term_index=0, count_index=1, encoding='utf8')

        
    def predict(self, array):
        array = self.processor(array, sampling_rate=16000).input_values[0]
        results = self.session.run(["output"],{"input":array.reshape(1,-1)})
        return np.argmax(results[0],axis=-1)
        
    
    def decode(self, sequence):
        pred_str = [self.index_to_word[idx] for idx in sequence.flatten()]
        remove_pad_token = re.sub('<pad>','',''.join(pred_str))
        ctc = []
        tmp = ""
        for s in remove_pad_token:
            if s == '|':
                s = " "
            if s == tmp:
                continue
            else:
                ctc.append(s)
            tmp = s
        suggestion = self.sym_spell.lookup_compound(
            "".join(ctc).strip(), max_edit_distance=2)
        return join_jamos(suggestion[0].term)
    
    def one_shot(self, array):
        pred = self.predict(array)
        return self.decode(pred)
    

if __name__ == '__main__':
    import librosa
    
    service = ServiceModel()
    array,_ = librosa.load('./Assets/test_data_1.wav',16000)
    print(type(array))
    print(array.shape)
    print(array.dtype)
    
    print('onnx :',service.one_shot(array))