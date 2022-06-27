import numpy as np
import librosa
import sounddevice as sd
from queue import Queue
import time
import sys
from scipy import signal

from model import ServiceModel

sr = 16000
ch = 1

global running, threshold
running = True
sig_flag = False

q = Queue()

def rms(array):
    return np.sqrt( np.mean( np.array(array) **2 ) )

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())
    
### new
    
ft = signal.firwin(1500,[300,3000],pass_zero=False,fs=sr)

def freq_filter(array):
    if len(array) > 2000:
        return signal.lfilter(ft,[1.0],array)
    else:
        return None
   
###

def get_q():
    array = []
    while not q.empty():
        array.extend(q.get())
    if len(array) > 1:
        return np.concatenate(array)
    else:
        return np.array(array)
    
model = ServiceModel()

def service(sig):
    print(model.one_shot(np.array(sig).flatten()))
    
    
if __name__ == "__main__":
    pause_cnt = 10
    sig_flag = False
    speech_sig = np.zeros(1,dtype=np.float32)
    try:
        print("init start - please be quiet")
        with sd.InputStream(samplerate=sr, channels=ch, callback=callback):
            time.sleep(2)
        sig = get_q()
        # print(sig.shape, sig,dtype)
        sig = freq_filter(sig) ### change
        
        # 시그마 규칙
        sigma_rule_num = 3
        sig = np.abs(sig)
        threshold = np.min(sig) + (np.std(sig) * sigma_rule_num )
        
        print("init complete")
        print("threshold :",threshold)
        with sd.InputStream(samplerate=sr, channels=ch, callback=callback):
            sig = np.zeros(1,dtype=np.float32)
            # print(sig.shape, sig,dtype)
            print("running start")
            
            while running:
                # time.sleep(0.05)
                sig = np.concatenate((sig, get_q()))
                harmonic_sig = freq_filter(sig) ### change
                
                if harmonic_sig is None:
                    continue
                
                if rms(harmonic_sig) > threshold:
                    print("saying")
                    # time.sleep(1)
                    speech_sig = np.concatenate((speech_sig,sig))
                    sig_flag = True
                    cnt = 0
                else:
                    if sig_flag:
                        if cnt < pause_cnt:
                            speech_sig = np.concatenate((speech_sig,sig))
                            cnt += 1
                        else:
                            print("STT")
                            service(speech_sig)
                            speech_sig = np.zeros(1,dtype=np.float32)
                            sig_flag = False

                sig = np.zeros(1,dtype=np.float32)
                
    except KeyboardInterrupt as ke:
        print("Recording finished")
        
    print("Program end")