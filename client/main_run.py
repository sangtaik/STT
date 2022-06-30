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
# import speech_recognition as sp_rec
import sounddevice as sd
from queue import Queue
from scipy import signal # NumPy와 Scipy를 함께 사용하면 확장 애드온을 포함한 MATLAB을 대체

# 파워포인트 실행 때문에 필요
import os
import time
import re

# 키보드 이벤트 강제 실행
from pynput.keyboard import Key, Controller

# global running, threshold
# running = True
sig_flag = False
sr = 16000
q = Queue()

def rms(array):
    return np.sqrt( np.mean( np.array(array) **2 ) )

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    q.put(indata.copy())


# import librosa   
# def hpss(array):
#     if len(array) > 2048:
#         stft = librosa.stft(array)
#         hamonic, _ = librosa.decompose.hpss(stft)
#         return librosa.istft(hamonic)
#     else:
#         return None
"""
2022.06.23 hpss 함수 -> signal 및 freq_filter로 개선함
"""
ft = signal.firwin(1500,[300,3000],pass_zero=False,fs=sr)

def freq_filter(array):
    if len(array) > 1600:
        return signal.lfilter(ft,[1.0],array)
    else:
        return None
    
def get_q():
    array = []
    while not q.empty():
        array.extend(q.get())
    if len(array) > 1:
        return np.concatenate(array)
    else:
        return np.array(array)


class VoiceWorker(QtCore.QObject):
    textChanged = QtCore.pyqtSignal(str)
    messageChanged = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    
    def __init__(self, parent=None):
        super(VoiceWorker, self).__init__(parent)
        self.model = ServiceModel()
        self.is_running = False
        
    def emit_message(self, msg):
        self.messageChanged.emit(msg)
        print(msg)
    
        
    def service(self, sig):
#         return self.model.one_shot(np.array(sig).flatten().astype(np.float32))
        return self.model.one_shot(np.array(sig).flatten())

#     @QtCore.pyqtSlot()
#     def task_old(self):
#         self.is_running = True
#         rec = sp_rec.Recognizer()
#         mic = sp_rec.Microphone()
#         sr = 16000
        
#         QtCore.QThread.sleep(1)
        
#         with mic as source: rec.adjust_for_ambient_noise(source)
#         while self.is_running == True:
#             print("Say somethig!")
#             with mic as source:
#                 sig = rec.listen(source).get_wav_data(sr,2)
#                 data_s16 = np.frombuffer(sig, dtype=np.int16, count=len(sig)//2)
#                 float_data = (data_s16 * 0.5**15).astype(np.float32)
#                 print("Got it! Now to recognize it...")
#                 try:
#                     # 모델에서 판별한 문장이 list로 출력된다.
#                     text_list = self.model.one_shot(np.array(float_data).flatten())

#                     text = ''.join(text_list)   # 나오는 것이 list이기 때문에 문장열로 만들어준다.
#                     self.textChanged.emit(text) # text를 textChanged로 정의한 함수에 return해준다.
#                     print(f"You said: {text}")
#                 except sp_rec.UnknownValueError:
#                     print(f"Oops, UnknownValueError : {sp_rec.UnknownValueError}")
#                     break
#                 time.sleep(1)
#         print("task done and stopped")
#         self.finished.emit()  # 예기치 못한 이유로 종료될 때에도 finished event를 호출
  

    """
    해당 task_old 함수는 기능은 되지만 인식률 및 음성 활동 감지(영어: Voice Activity Detection 또는 VAD)에 차이가 나서 
    성능을 향상을 이유로 로직을 변경함
    """
    @QtCore.pyqtSlot()
    def task(self):
        self.is_running = True
        sr = 16000
        ch = 1
        
        sigma_rule_num = 4  # sigma 규칙에 표준편차를 늘려주는 역할
        pause_cnt = 5
        sig_flag = False
        speech_sig = np.zeros(1,dtype=np.float32)
        
        try:
            with sd.InputStream(samplerate=sr, channels=ch, callback=callback):
                self.emit_message("음성 인식 환경을 초기화합니다. 2초만 조용히 해주세요.")
                time.sleep(2)
            """
                마이크 입력을 받아서 음성 인식 환경을 초기화
                통계학에서 68-95-99.7 규칙은 정규 분포를 나타내는 규칙으로, 
                경험적인 규칙(empirical rule)이라고도 한다. 
                3시그마 규칙(three-sigma rule)이라고도 하는데 
                이 때는 평균에서 양쪽으로 3표준편차의 범위에 거의 모든 값들(99.7%)이 들어간다는 것을 나타낸다.
            """
            print("voice init start")
            sig = get_q()
            print(sig.dtype)
            sig = freq_filter(sig) # hpss대신 사용

            ## 시그마 규칙 : 통계 부분에 나온다.
            # 환경음의 평균 소리를 진폭을 +, -값을 절대값으로 만들어줘서 위상을 없앤다.
            sig = np.abs(sig) 
            
            # 환경음의 최대값 어디까지인지 구한다.
            # 이 최대값을 벗어나면 사람이 말하는 음성으로 인식할 것이다.
            # 2022-06-28  np.min -> mean으로 변경, 멀리있는 음성 인식은 손해가 되지만 가까운 음성 잡음에 강하게 될 것으로 기대함
            threshold = np.mean(sig) + (np.std(sig) * sigma_rule_num ) 
            
            print("voice init complete")
            
            
            msg = "threshold 설정값: {}".format(threshold)
            self.emit_message(msg)
            self.emit_message("음성 인식 환경을 초기화를 완료화였습니다.")
            with sd.InputStream(samplerate=sr, channels=ch, callback=callback):
                sig = np.zeros(1,dtype=np.float32)
                tmp = np.zeros(1,dtype=np.float32)
                print(sig.dtype)
                print("running start")
                self.emit_message("명령을 말씀해주세요.")
                while self.is_running == True:

                    sig = np.concatenate((sig, get_q()))

                    harmonic_sig = freq_filter(sig)
                    if harmonic_sig is None:
                        continue

                    if rms(harmonic_sig) > threshold:
                        print("saying")
                        start_time = time.time()
                        # time.sleep(1)
                        speech_sig = np.concatenate((speech_sig,tmp))
                        sig_flag = True
                        cnt = 0

                    else:
                        if sig_flag:
                            if cnt < pause_cnt:
                                speech_sig = np.concatenate((speech_sig,tmp))
                                cnt += 1
                            else:
                                if len(speech_sig) < sr*7: # 7초 이상 음성은 받지 않는다.
                                    print("STT")
                                    print(time.time()-start_time)
                                     # 모델에서 판별한 문장이 list로 출력된다.
                                    self.emit_message("명령을 분석하고 있습니다.")
                                    print("Got it! Now to recognize it...")
                                    text_list = self.service(speech_sig)

                                    text = ''.join(text_list)   # 나오는 것이 list이기 때문에 문장열로 만들어준다.
                                    self.textChanged.emit(text) # text를 textChanged로 정의한 함수에 return해준다.
                                    msg = "You said: {}".format(text)
                                    self.emit_message(msg)
                                    speech_sig = np.zeros(1,dtype=np.float32)
                                    sig_flag = False
                                    self.emit_message("명령을 말씀해주세요.")
                                else:
                                    self.emit_message("긴 대기시간으로 다음 음성을 입력받습니다.")
                                    speech_sig = np.zeros(1,dtype=np.float32)
                                    sig_flag = False

                    tmp = sig
                    sig = np.zeros(1,dtype=np.float32)

        except Exception as e:
                print(f"Oops, UnknownError : {str(e)}")
        time.sleep(1)
        print("task done and stopped")
        self.emit_message("음성 인식 작업을 중단합니다.")
        self.finished.emit()  # 예기치 못한 이유로 종료될 때에도 finished event를 호출

    def stop(self):
        self.is_running = False

    
class ExampleApp(QtWidgets.QMainWindow, ui_main.Ui_MainWindow):
    
    def __init__(self, parent=None):
        pyqtgraph.setConfigOption('background', 'w') #before loading widget
        super(ExampleApp, self).__init__(parent)
        self.setupUi(self)
        
        # 마이크 테스트 UI : 음성이 마이크에 제대로 도달하는지 확인하는 UI
        self.isMikeTest = True
        self.grPCM.plotItem.showGrid(True, True, 1)
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
        self.test_button.clicked.connect(self.do_list_command_test)
        self.worker.textChanged.connect(self.do_list_command)
        self.worker.messageChanged.connect(self.append_message)
        self.worker.finished.connect(self.handleFinished)
        
        # 버튼 활성화 상태를 지정 : Start 버튼을 활성화.
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
       
        
        
        self.test_list = ['피피티 시작', '이에스씨 눌러', '슬라이드쇼 시작', '다음 페이지'
           , '이전 페이지', '3번째 페이지', '처음페이지로', '마지막페이지', '슬라이드쇼 종료', '피피티 종료'] 
        
        
#         self.cmd_list = ['피피티 시작(켜죠)', '피피티 종료(꺼죠)', '슬라이드쇼 시작(켜죠)', '슬라이드쇼 종료(끝)', '다음(뒤) 페이지(장)'
#            , '이전(앞) 페이지(장)', '처음(첫, 시작) 페이지(장)', '마지막(끝) 페이지(장)', '이에스씨(창닫기) 실행(눌러)'
                         
        self.cmd_list = ['피피티 시작(켜줘)'
#                          , '피피티 종료(꺼줘)' # 발음이 서로 인접된 문제때문에 켜줘와 꺼줘 구분이 잘 되지 않는다.
                         , '슬라이드쇼 시작(열어)', '슬라이드쇼 종료(끝)', '다음(뒤) 페이지'
           , '이전(앞) 페이지', '처음(첫, 시작) 페이지', '마지막(끝) 페이지', '이에스씨(창닫기) 실행(눌러)'
#                          , '<숫자>페이지(장) 이동(열어)'
                        ] 
        self.reg_list = [ '(?=(.*피피티|파워포인트.*){1,})(?=(.*시작|.*열어.*){1,}).*'
#            , '(?=(.*피피티|파워포인트.*){1,})(?=(.*종료|.*끝|.*꺼.*){1,}).*'
           , ''
           , '(?=(.*슬라이드.*|.*슬라이드쇼.*|.*발표.*){1,})(?=(.*시작|열어,*){1,}).*'
           , '(?=(.*슬라이드.*|.*슬라이드쇼.*|.*발표.*){1,})(?=(.*종료|끝,*){1,}).*'
           , '(?=(.*다음.*|.*뒤.*){1,})(?=(.*페이지){1,}).*'
           , '(?=(.*이전.*|.*앞.*){1,})(?=(.*페이지){1,}).*'
           , '(?=(.*처음.*|.*첫.*|.*시작.*){1,})(?=(.*페이지){1,}).*'
           , '(?=(.*마지막.*|.*끝.*){1,})(?=(.*페이지){1,}).*'
           , '(?=(.*이에스씨.*|.*창닫기.*){1,})(?=(.*눌러|.*실행){1,}).*']
#            , '(?=(.*[0-9]페이지.*|.*[0-9]장.*){1,})(?=(.*이동|.*열어){1,}).*'  # 추후 필요 시 구현할 기능

    def handleStarted(self):
        self.isMikeTest = False
        print('handleStarted')
        self.thread.start()                 # thread 재시작 : 이미 event로 연동되어 있는 VoiceWorker.task 함수를 실행한다.
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.label_2.setText("Message Log")
        self.grPCM.setHidden(True)
        self.textMessage.setHidden(False)
        print('handleStarted finished')
                        
    def handleFinished(self):
        self.isMikeTest = True
        print('handleFinished fuc start')
        self.worker.stop()
        print('worker.stop')
        self.thread.quit()
        print('thread.quited')
        self.thread.wait()
        print('thread.waited')
        self.label_2.setText("raw data (PCM")
        self.grPCM.setHidden(False)
        self.textMessage.setHidden(True)
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
        
    def count_test_lines(self):
        text = self.textBrowser.toPlainText()
        num_lines = len(text.split('\n'));
        return num_lines
    
    def count_message_lines(self):
        text = self.textMessage.toPlainText()
        num_lines = len(text.split('\n'));
        return num_lines
        
    def append_text(self, text):
        num_lines = self.count_test_lines()
        if  num_lines > 20:
            self.textBrowser.clear()
        
        self.textBrowser.append(text)
        
    def append_message(self, text):
        num_lines = self.count_message_lines()
        if  num_lines > 20:
            self.textMessage.clear()
        
        self.textMessage.append(text)
        
        
    def do_list_command(self, text):
        self.append_text(text)
      
        keyboard = Controller()

        if re.match(self.reg_list[0], text) is not None:
            cmd = 'test.pptx'
            self.open_program(cmd)
            time.sleep(5) # wait for 5 sec
            print("key press : esc")  # 오피스365 가입하라는 창을 제거하기 위해서
            keyboard.press(Key.esc)
            print(f"\t{self.reg_list[0]}: 실행완료")
#         elif re.match(self.reg_list[1], text) is not None:
#             # send ALT+F4 in the same time, and then send space, 
#             # (be carful, this will close any current open window)
#             keyboard.press(Key.alt)
#             keyboard.press(Key.f4) 
#             keyboard.release(Key.alt)
#             keyboard.release(Key.f4)
#             print(f"\t{self.reg_list[1]}: 실행완료")
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


        

    def do_list_command_test(self, event):
                
        keyboard = Controller()
        for text in self.test_list:
            print(text)
            time.sleep(2) # wait for [N] sec
            self.do_list_command(text)
            
        
    
    def update(self):
        if self.isMikeTest == True:
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

    




