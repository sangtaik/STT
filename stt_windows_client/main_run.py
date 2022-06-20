#!/usr/bin/env python
# coding: utf-8

# STT의 공방은 https://github.com/anuragseven/azurespeech-pyqt5/blob/main/speech.py를 참고
# 원하는 event를 생성하는 것은 링크 참조 https://wikidocs.net/21876
# 쓰레드 사용 및 signal은 링크 참조
# https://stackoverflow.com/questions/56200533/when-i-try-to-run-speech-recognition-in-pyqt5-program-is-crashed

from PyQt5 import QtGui, QtCore, QtWidgets

import sys
import ui_main
import numpy as np
import pyqtgraph
import datetime
import SWHear

# model
from model import ServiceModel
import speech_recognition as sp_rec

class VoiceWorker(QtCore.QObject):
    textChanged = QtCore.pyqtSignal(str)
    
    def __init__(self, parent=None):
        super(VoiceWorker, self).__init__(parent)
        self.model = ServiceModel()  

    
    @QtCore.pyqtSlot()
    def task(self):
        rec = sp_rec.Recognizer()
        mic = sp_rec.Microphone()
        sr = 16000

        while True:
            print("Say somethig!")
            with mic as source:
                sig = rec.listen(source).get_wav_data(sr,2)
                data_s16 = np.frombuffer(sig, dtype=np.int16, count=len(sig)//2)
                float_data = (data_s16 * 0.5**15).astype(np.float32)
                print("Got it! Now to recognize it...")
                try:
                    # 모델에서 판별한 문장이 list로 출력된다.
                    text_list = self.model.one_shot(np.array(float_data).flatten())
                    
                    # 나오는 것이 list이기 때문에 문장열로 만들어준다.
                    text = ''.join(text_list)
                    self.textChanged.emit(text)
                    print("You said: {}".format(text))
                except sp_rec.UnknownValueError:
                    print("Oops")
     
    
    
class ExampleApp(QtWidgets.QMainWindow, ui_main.Ui_MainWindow):
    
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.grPCM.plotItem.showGrid(True, True, 0.5)
        self.maxPCM=0
        self.ear = SWHear.SWHear(rate=44100,updatesPerSecond=20)
        self.ear.stream_start()

        
        self.worker = VoiceWorker()
        self.thread = QtCore.QThread()
        self.thread.start()
        
        self.worker.moveToThread(self.thread)
        self.start_button.clicked.connect(self.worker.task)
        self.close_button.clicked.connect(self.closeEvent)
        self.worker.textChanged.connect(self.textBrowser.setText)
        
    def update(self):
#         if not self.ear.data is None and not self.ear.fft is None:
#             pcmMax=np.max(np.abs(self.ear.data))
#             if pcmMax>self.maxPCM:
#                 self.maxPCM=pcmMax
#                 self.grPCM.plotItem.setRange(yRange=[-pcmMax,pcmMax])
#             self.pbLevel.setValue(1000*pcmMax/self.maxPCM)
#             pen=pyqtgraph.mkPen(color='b')
#             self.grPCM.plot(self.ear.datax ,self.ear.data ,pen=pen ,clear=True)
#             pen=pyqtgraph.mkPen(color='r')
  
        # 라벨에 텍스트 표시
        current_time = str(datetime.datetime.now().time())
        self.label_time.setText(current_time)
        self.label_time.repaint()
        
        QtCore.QTimer.singleShot(1, self.update) # QUICKLY repeat
        
    # 윈도우 종료
    def closeEvent(self, event):
        event.accept()

    def keyPressEvent(self, event):
        """Close application from escape key.

        results in QMessageBox dialog from closeEvent, good but how/why?
        """
        if event.key() == Qt.Key_Escape:
            self.close()
        
        
if __name__=="__main__":
    print('start app')
    app = QtWidgets.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    form.update() #start with something
    app.exec_()
    print("close app")
    




