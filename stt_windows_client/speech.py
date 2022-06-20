#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PyQt5.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal
import azure.cognitiveservices.speech as speechsdk
from playsound import playsound

API_KEY = ""
REGION = ""


class Stt(QRunnable):
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()

    def run(self) -> None:
        speech_config = speechsdk.SpeechConfig(subscription=API_KEY, region=REGION)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)
#         playsound('music/StartBeep.wav')
        result = speech_recognizer.recognize_once()
#         playsound("music/EndBeep.wav")

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self.signals.finished.emit(result.text)
        elif result.reason == speechsdk.ResultReason.NoMatch:
            print("No speech could be recognized")
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print("Speech Recognition canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))

    def start(self):
        QThreadPool.globalInstance().start(self)
        
class WorkerSignals(QObject):
    finished = pyqtSignal(str)

