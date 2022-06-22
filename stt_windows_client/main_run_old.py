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

# pcm 표시
import SWHear

# model
from model import ServiceModel
import speech_recognition as sp_rec

# 파워포인트 실행 때문에 필요
import os
import time
# import asyncio

# 키보드 이벤트 강제 실행
from pynput.keyboard import Key, Controller


class VoiceWorker(QtCore.QObject):
    textChanged = QtCore.pyqtSignal(str)
    
    def __init__(self, parent=None):
        super(VoiceWorker, self).__init__(parent)
        self.model = ServiceModel()
        # mutex 설정
        self.mutex = QtCore.QMutex()  # 뮤텍스로 명령어가 여러가지 중첩 실행하는 것을 막고 1 by 1으로 실행하도록 관리한다.
        
    
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

                    text = ''.join(text_list)   # 나오는 것이 list이기 때문에 문장열로 만들어준다.
                    self.textChanged.emit(text) # text를 textChanged로 정의한 함수에 return해준다.
                    print("You said: {}".format(text))
                except sp_rec.UnknownValueError:
                    print("Oops, UnknownValueError")
            time.sleep(1)

            

    
class ExampleApp(QtWidgets.QMainWindow, ui_main.Ui_MainWindow):
    
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        self.grPCM.plotItem.showGrid(True, True, 0.5)
        self.maxPCM=0
        self.ear = SWHear.SWHear(rate=44100,updatesPerSecond=20)
        self.ear.stream_start()
        
        # 모델을 실행할 thread 생성
        self.worker = VoiceWorker()
        self.thread = QtCore.QThread()
        self.thread.start()
        
        self.worker.moveToThread(self.thread) # thread를 worker에 이전
        self.start_button.clicked.connect(self.worker.task)
        self.close_button.clicked.connect(self.do_list_event)
        self.worker.textChanged.connect(self.do_list)
        
    def open_program(self, cmd):
        print("{} starting now", cmd)
        # 이슈 : os.system(cmd)은 파일이 열린 후에도 아래 코드가 실행이 안된다.
        # 오픈한 프로그램을 닫아야지만 다음 줄 코드가 실행이 된다.
        # 따라서 async를 사용해서 시도해봤으나, 해당 함수가 계속 메인 프로시저를 , 해결되지 않는다.
        # 결과적으로 startfile 함수를 사용하여 실행만 하고, 다음으로 넘어간다.
        os.startfile(cmd) 
        print("{} starting complete", cmd)
        
        
    def do_list(self, text):
        self.textBrowser.setText(text)
      
        keyboard = Controller()

        if text == "피피티 시작":
            cmd = 'test.pptx'
            self.open_program(cmd)
            time.sleep(5) # wait for 5 sec
            print("key press : esc")  # 오피스365 가입하라는 창을 제거하기 위해서
            keyboard.press(Key.esc)
        elif text == "피피티 종료":
            # send ALT+F4 in the same time, and then send space, 
            # (be carful, this will close any current open window)
            keyboard.send("alt+F4, space")  
        elif text == "슬라이드쇼 시작":
            keyboard.press(Key.f5)
            keyboard.release(Key.f5)
        elif text == "슬라이드쇼 종료":
            keyboard.press(Key.esc)
        elif text == "다음 페이지":
            keyboard.press(Key.right)
            keyboard.release(Key.right)
        elif text == "이전 페이지":
            keyboard.press(Key.left)
            keyboard.release(Key.left)
        elif text == "처음페이지":
            keyboard.press(Key.home)
        elif text == "마지막페이지":
            keyboard.press(Key.end)
        

    def do_list_event(self, event):
        
#         a_string = "A string is more than its parts!"
# matches = ["more", "wholesome", "milk"]

# if any(x in a_string for x in matches):
        
        cmd_list = ['피피티 시작', '슬라이드쇼 시작', '다음 페이지'
                    , '이전 페이지', '처음페이지로', '마지막페이지', '슬라이드쇼 종료', '피피티 종료']  
#         cmd_list = ['피피티 시작합니다.', '아 슬라이드쇼 시작합니다', '다음 페이지로'
#                     , '이전 페이지', '처음페이지로', '마지막페이지', '슬라이드쇼 종료', '피피티 종료']  
        keyboard = Controller()
        for text in cmd_list:
            print(text)
            time.sleep(1) # wait for 5 sec
            if text == "피피티 시작":
                    cmd = 'test.pptx'
                    self.open_program(cmd)
                    time.sleep(5) # wait for 5 sec
                    print("key press : esc")  # 오피스365 가입하라는 창을 제거하기 위해서
                    keyboard.press(Key.esc)
            elif text == "피피티 종료":
                # send ALT+F4 in the same time, and then send space, 
                # (be carful, this will close any current open window)
                keyboard.press(Key.alt)
                keyboard.press(Key.f4) 
                keyboard.release(Key.alt)
                keyboard.release(Key.f4)
            elif text == "슬라이드쇼 시작":
                keyboard.press(Key.f5)
                keyboard.release(Key.f5)
            elif text == "슬라이드쇼 종료":
                keyboard.press(Key.esc)
            elif text == "다음 페이지":
                keyboard.press(Key.right)
                keyboard.release(Key.right)
            elif text == "이전 페이지":
                keyboard.press(Key.left)
                keyboard.release(Key.left)
            elif text == "처음페이지":
                keyboard.press(Key.home)
            elif text == "마지막페이지":
                keyboard.press(Key.end)

        
    
    def update(self):
        if not self.ear.data is None and not self.ear.fft is None:
            pcmMax=np.max(np.abs(self.ear.data))
            if pcmMax>self.maxPCM:
                self.maxPCM=pcmMax
                self.grPCM.plotItem.setRange(yRange=[-pcmMax,pcmMax])
            self.pbLevel.setValue(1000*pcmMax/self.maxPCM)
            pen=pyqtgraph.mkPen(color='b')
            self.grPCM.plot(self.ear.datax ,self.ear.data ,pen=pen ,clear=True)
            pen=pyqtgraph.mkPen(color='r')
  
        # 라벨에 텍스트 표시
        current_time = str(datetime.datetime.now().time())
        self.label_time.setText(current_time)
        self.label_time.repaint()
        
        
        QtCore.QTimer.singleShot(1, self.update) # QUICKLY repeat
    
    def close(self):
        if self.thread is not None:
            self.thread.quit()
            print('thread was quited')
        self.ear.close()
        print("closed app")
        sys.exit(app.exec_())
        
    # 윈도우 종료
    def closeEvent(self, event):
        print('closeEvent')
        self.close()


    def keyPressEvent(self, event):
        if event.key() == Key.esc:
            print('keyPressEvent : escape')
            self.close()
        
        
if __name__=="__main__":
    print('start app')
    app = QtWidgets.QApplication.instance()
    if app is None: 
        app = QtWidgets.QApplication(sys.argv)
    form = ExampleApp()
    form.show()
    form.update() #start with something
    app.exec_()

    




