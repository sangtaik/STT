#!/usr/bin/env python
# coding: utf-8

# STT의 공방은 https://github.com/anuragseven/azurespeech-pyqt5/blob/main/speech.py를 참고
# 원하는 event를 생성하는 것은 링크 참조 https://wikidocs.net/21876
# 쓰레드 사용 및 signal은 링크 참조
# https://stackoverflow.com/questions/56200533/when-i-try-to-run-speech-recognition-in-pyqt5-program-is-crashed

from PyQt5 import QtGui, QtCore, QtWidgets

import sys

import numpy as np
import pyqtgraph
import datetime

# 화면 ui
import ui_main

# pcm 표시
import SWHear

# model 생성 및 사운드 입력
from model import ServiceModel
import speech_recognition as sp_rec

# 파워포인트 실행 때문에 필요
import os
import time
import re

# 키보드 이벤트 강제 실행
from pynput.keyboard import Key, Controller


class VoiceWorker(QtCore.QObject):
    textChanged = QtCore.pyqtSignal(str)
#     started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super(VoiceWorker, self).__init__(parent)
        self.model = ServiceModel()
        self.isRunning = False
    
    @QtCore.pyqtSlot()
    def task(self):
        self.isRunning = True
        rec = sp_rec.Recognizer()
        mic = sp_rec.Microphone()
        sr = 16000
        with mic as source: rec.adjust_for_ambient_noise(source)
        
        while self.isRunning == True:
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
                    print(f"You said: {text}")
                except sp_rec.UnknownValueError:
                    print(f"Oops, UnknownValueError : {sp_rec.UnknownValueError}")
                    break
            time.sleep(1)
        print("task done and stopped")
        self.finished.emit()
        
    def stop(self):
        self.isRunning = False

    
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
        
        self.worker.moveToThread(self.thread) # thread를 worker에 이전
        self.thread.started.connect(self.worker.task)
        
        self.start_button.clicked.connect(self.handleStarted)
        self.stop_button.clicked.connect(self.handleFinished)
        self.test_button.clicked.connect(self.do_list_event_test)
        self.worker.textChanged.connect(self.do_list)
#         self.worker.started.connect(self.handleStarted)
        self.worker.finished.connect(self.handleFinished)
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        
        self.test_list = ['피피티 시작', '이에스씨 눌러', '슬라이드쇼 시작', '다음 페이지'
           , '이전 페이지', '3번째 페이지', '처음페이지로', '마지막페이지', '슬라이드쇼 종료', '피피티 종료'] 
        self.cmd_list = ['피피티 시작', '피피티 종료', '슬라이드쇼 시작', '슬라이드쇼 종료(끝)', '다음(뒷) 페이지(장)'
           , '이전(앞) 페이지(장)', '처음(첫, 시작) 페이지(장)', '마지막(끝) 페이지(장)', '이에스씨(창닫기) 실행(눌러)'
#                          , '<숫자>페이지(장) 이동(열어)'
                        ] 
        self.reg_list = [ '(?=(.*피피티|파워포인트.*){1,})(?=(.*시작|.*열어){1,}).*'
           , '(?=(.*피피티|파워포인트.*){1,})(?=(.*종료|.*끝){1,}).*'
           , '(?=(.*슬라이드|슬라이드쇼.*){1,})(?=(.*시작|모드,*){1,}).*'
           , '(?=(.*슬라이드|슬라이드쇼.*){1,})(?=(.*종료|끝,*){1,}).*'
           , '(?=(.*다음.*|.*뒷.*){1,})(?=(.*페이지|.*장){1,}).*'
           , '(?=(.*앞.*|.*이전.*){1,})(?=(.*페이지|.*장){1,}).*'
           , '(?=(.*처음.*|.*첫.*|.*시작.*){1,})(?=(.*페이지|.*장){1,}).*'
           , '(?=(.*마지막.*|.*끝.*){1,})(?=(.*페이지|.*장){1,}).*'
           , '(?=(.*이에스씨.*|.*창닫기.*){1,})(?=(.*눌러|.*실행){1,}).*']
#            , '(?=(.*[0-9]페이지.*|.*[0-9]장.*){1,})(?=(.*이동|.*열어){1,}).*'

    def handleStarted(self):
        print('handleStarted')
#         self.worker.moveToThread(self.thread) # thread를 worker에 이전
#         self.thread.started.connect(self.worker.task)
        self.thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        print('handleStarted finished')
                        
    def handleFinished(self):
        print('handleFinished fuc start')
        self.worker.stop()
        print('worker.stop')
        self.thread.quit()
        print('thread.quited')
        self.thread.wait()
        print('thread.waited')
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        print('handleFinished fuc end')
        
    def open_program(self, cmd):
        print("{cmd} starting now")
        # 이슈 : os.system(cmd)은 파일이 열린 후에도 아래 코드가 실행이 안된다.
        # 오픈한 프로그램을 닫아야지만 다음 줄 코드가 실행이 된다.
        # 따라서 async를 사용해서 시도해봤으나, 해당 함수가 계속 메인 프로시저를 , 해결되지 않는다.
        # 결과적으로 startfile 함수를 사용하여 실행만 하고, 다음으로 넘어간다.
        os.startfile(cmd) 
        print("{cmd} starting complete")
        
        
    def do_list(self, text):
        
        self.textBrowser.setText(text)
      
        keyboard = Controller()

        if re.match(self.reg_list[0], text) is not None:
            cmd = 'test.pptx'
            self.open_program(cmd)
            time.sleep(5) # wait for 5 sec
            print("key press : esc")  # 오피스365 가입하라는 창을 제거하기 위해서
            keyboard.press(Key.esc)
            print(f"\t{self.reg_list[0]}: 실행완료")
        elif re.match(self.reg_list[1], text) is not None:
            # send ALT+F4 in the same time, and then send space, 
            # (be carful, this will close any current open window)
            keyboard.press(Key.alt)
            keyboard.press(Key.f4) 
            keyboard.release(Key.alt)
            keyboard.release(Key.f4)
            print(f"\t{self.reg_list[1]}: 실행완료")
        elif re.match(self.reg_list[2], text) is not None:
            keyboard.press(Key.f5)
            keyboard.release(Key.f5)
            print(f"\t{self.reg_list[2]}: 실행완료")
        elif re.match(self.reg_list[3], text) is not None:
            keyboard.press(Key.esc)
            print(f"\t{self.reg_list[3]}: 실행완료")
        elif re.match(self.reg_list[4], text) is not None:
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            print(f"\t{self.reg_list[4]}: 실행완료")
        elif re.match(self.reg_list[5], text) is not None:
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            print(f"\t{self.reg_list[5]}: 실행완료")
        elif re.match(self.reg_list[6], text) is not None:
            keyboard.press(Key.home)
            print(f"\t{self.reg_list[6]}: 실행완료")
        elif re.match(self.reg_list[7], text) is not None:
            keyboard.press(Key.end)
            keyboard.release(Key.end)
            print(f"\t{self.reg_list[7]}: 실행완료")
        elif re.match(self.reg_list[8], text) is not None:
            keyboard.press(Key.esc)
            print(f"\t{self.reg_list[8]}: 실행완료")
#         elif re.match(self.reg_list[9], text) is not None:
#             keyboard.press(Key.ctrl)
#             keyboard.press(Key.f5) 
#             keyboard.release(Key.ctrl)
#             keyboard.release(Key.f5)
#             print(f"\t{self.reg_list[9]}: 실행완료")
        else:
            print(f"{text}: 실행목록이 없습니다.")
            print(f"실행목록 예시 : {self.cmd_list}")


        

    def do_list_event_test(self, event):
                
        keyboard = Controller()
        for text in self.test_list:
            print(text)
            time.sleep(2) # wait for [N] sec
            if re.match(self.reg_list[0], text) is not None:
                cmd = 'test.pptx'
                self.open_program(cmd)
                time.sleep(5) # wait for 5 sec
                print("key press : esc")  # 오피스365 가입하라는 창을 제거하기 위해서
                keyboard.press(Key.esc)
                print(f"\t{self.reg_list[0]}: 실행완료")
            elif re.match(self.reg_list[1], text) is not None:
                # send ALT+F4 in the same time, and then send space, 
                # (be carful, this will close any current open window)
                keyboard.press(Key.alt)
                keyboard.press(Key.f4) 
                keyboard.release(Key.alt)
                keyboard.release(Key.f4)
                print(f"\t{self.reg_list[1]}: 실행완료")
            elif re.match(self.reg_list[2], text) is not None:
                keyboard.press(Key.f5)
                keyboard.release(Key.f5)
                print(f"\t{self.reg_list[2]}: 실행완료")
            elif re.match(self.reg_list[3], text) is not None:
                keyboard.press(Key.esc)
                print(f"\t{self.reg_list[3]}: 실행완료")
            elif re.match(self.reg_list[4], text) is not None:
                keyboard.press(Key.right)
                keyboard.release(Key.right)
                print(f"\t{self.reg_list[4]}: 실행완료")
            elif re.match(self.reg_list[5], text) is not None:
                keyboard.press(Key.left)
                keyboard.release(Key.left)
                print(f"\t{self.reg_list[5]}: 실행완료")
            elif re.match(self.reg_list[6], text) is not None:
                keyboard.press(Key.home)
                print(f"\t{self.reg_list[6]}: 실행완료")
            elif re.match(self.reg_list[7], text) is not None:
                keyboard.press(Key.end)
                keyboard.release(Key.end)
                print(f"\t{self.reg_list[7]}: 실행완료")
            elif re.match(self.reg_list[8], text) is not None:
                keyboard.press(Key.esc)
                print(f"\t{self.reg_list[8]}: 실행완료")
#             elif re.match(self.reg_list[9], text) is not None:
#                 num = 0
#                 keyboard.press(Key.num)
#                 keyboard.press(Key.enter) 
#                 keyboard.release(Key.ctrl)
#                 keyboard.release(Key.enter)
#                 print(f"\t{self.reg_list[9]}: 실행완료")
            else:
                print(f"{text}: 실행목록이 없습니다.")
                print(f"실행목록 예시 : {self.cmd_list}")
        
    
    def update(self):
        """
        if not self.ear.data is None and not self.ear.fft is None:
            pcmMax=np.max(np.abs(self.ear.data))
            if pcmMax>self.maxPCM:
                self.maxPCM=pcmMax
                self.grPCM.plotItem.setRange(yRange=[-pcmMax,pcmMax])
            self.pbLevel.setValue(1000*pcmMax/self.maxPCM)
            pen=pyqtgraph.mkPen(color='b')
            self.grPCM.plot(self.ear.datax ,self.ear.data ,pen=pen ,clear=True)
            pen=pyqtgraph.mkPen(color='r')
        """
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

    




