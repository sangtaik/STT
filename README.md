## PPT 도우미:   Speech To Text(STT) AI를 통하여 음성을 텍스트로 변환 기록하고 음성으로 PPT 슬라이드 쇼를 위한 기능을 구현한다.(영우글로벌러닝 AI 6기 Final Project)

[PPT Speech To Text Assistant : Korean Goal-Oriented Dialog Speech Corpus for Pretrained Speech Recognition of PowerPoint](https://github.com/sangtaik/STT)

[Sang-Taik Jung](https://github.com/sangtaik/STT)<sup>1,2</sup>, [Mu-Jun Kim](https://github.com/Mu-jun/STT)<sup>1,3</sup>, [Jae-Youn Park](https://github.com/jayo9901/)<sup>1,4</sup>

<sup>1</sup> K-Digital Training - AI Youngwoo Global Training AI 6th. <p>
<sup>2</sup> Client, Dataset preprocessing <p>
<sup>3</sup> AI Modeling, AI Training, Audio Signal Processing <p>
<sup>4</sup> Model Training Calibration, Dataset preprocessing.


## Table of contents 

* [1. Dataset contribution](#1-Dataset-contribution)
    + [The dataset statistics](#the-dataset-statistics)
    + [The dataset structure](#the-dataset-structure)
* [2. Dataset downloading and license](#2-dataset-downloading-and-license)
    + [AI Hub](#aihub)
* [3. Model](#3-model)
* [4. Dependency](#4-dependency)
* [5. Training and Evaluation](#5-training-and-evaluation)
    + [Run create csv for dateset preprocessing](#run-create-csv-for-dateset-preprocessing)
    + [Run vocab](#run-vocab)
    + [Run vocab](#run-vocab)
    + [Run train](#run-train)
    + [Run Client ](#run-client)
* [6. pyinstaller](#6-pyinstaller-for-exe-file))
* [7. Reference](#7-reference)



## 1. Dataset contribution
데이터 셋을 수집하기 위해서 AI HUB 한국어 자료 및 직접 녹음을 하여 데이터셋으로 가공함

오디오 파일형식 : PCM, WAV
텍스트 파일형식 : txt, json
최종 데이터 셋 파일형식 : csv

### The dataset statistics
명령어 음성(일반남녀) https://aihub.or.kr/aidata/30707 <p>
한국어 음성 https://aihub.or.kr/aidata/105 <p>
한국어 강의 음성 https://aihub.or.kr/aidata/30708 <p>


### The dataset structure
아래와 같이 음성 파일명, 대사, 음성데이터 값을 csv 파일로 추출하여 최종 훈련 데이터셋으로 가공하였다.
음성 파일 및 텍스트 파일이 있다면 전처리 폴더에 있는 XLSR_Wav2Vec2_colab_recursive_vocab_jamo.py 파일을 이용하여 추출할 수 있다.

We export the csv file for Training with the following structure by some export folder .py file: <p>
For Example :  XLSR_Wav2Vec2_colab_recursive_vocab_jamo.py <p>
```
order_speech_ko1000_000.csv
...
script1_g_0044-6003-01-01-KSM-F-05-A.wav	나 대신 점등해 줘.	[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.6443534e-13, ...]
script1_g_0044-6004-01-01-KSM-F-05-A.wav	이번 주 대체로 흐린지 궁금해.	[3.2407155e-07, -3.7628763e-07, 4.2549107e-07, ...]
```



## 2. Dataset downloading and license

1.  자료의 정확성, 완전성, 무결성, 품질 또는 적절성을 보증하지 않으므로 여기에 제공된 자료에 대해 책임을 지지 않습니다.
2.  각 데이터 셋 자료의 라이선스 및 사용권에 대하여는 AI HUB(https://aihub.or.kr/aidata)를 참조하시고, 문의하시기 바랍니다.

```
명령어 음성(일반남녀) [link] (https://aihub.or.kr/aidata/30707)
한국어 음성 [link] (https://aihub.or.kr/aidata/105)
한국어 강의 음성 [link] (https://aihub.or.kr/aidata/30708)
```

1. we DOES NOT GUARANTEE THE ACCURACY, COMPLETENESS, INTEGRITY, QUALITY OR ADEQUACY OF THE MATERIALS, THUS ARE NOT LIABLE OR RESPONSIBLE FOR THE MATERIALS PROVIDED HERE.
2. For information on licenses and usage rights for each data set, please refer to the AI ​​HUB (https://aihub.or.kr/aidata)


`AIhub` dataset can be download from 
```
KSM [here] (https://aihub.or.kr/aidata/30707)
KsponSpeech[here] (https://aihub.or.kr/aidata/105)
KlecSpeech [here] (https://aihub.or.kr/aidata/30708)
```
(`AIhub` : this is a large-scale Korean open domain dialog corpus from NIA AIHub5, an open data hub site of Korean Govern-ment.) 



## 3. Model

Wav2Vec 모델(wav2vec 2.0 - A Framework for Self-Supervised)을 사용하였으며, 자모 기반 단어장을 기반으로 Training한다.
명령어 목록에 편향된 사용자 음성을 추가하여 트레이닝한다.

모델에 대한 자세한 내용은 [여기](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)를 참조한다.


The Wav2Vec model(wav2vec 2.0 - A Framework for Self-Supervised) is used, and it is trained by analyzing morphemes by dividing it into Jamo.
Train by adding biased user voices to the command list.

[Model Link] (https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)


## 4. Dependency

Our code requires the following libraries:

```
librosa==0.8.1
matplotlib==3.2.2
numpy==1.21.6
pandas==1.3.5
requests==2.23.0
scipy==1.4.1
sklearn==0.0
SoundFile==0.10.3.post1
tensorboard==2.8.0
tensorflow==2.8.2
tensorflow-hub==0.12.0
tqdm==4.64.0
xgboost==0.90
jupyter==1.0.0
datasets==2.2.2
transformers==4.19.4
jiwer==2.30
SpeechRecognition==3.8.1
pipwin==0.5.2
```

## 5. Training and Evaluation

Before training or evaluation, we should be follow the data pipeline as the followed.

### Run create csv for dateset preprocessing

```
cd data_preprocessing
mkdir /dataset/csv/
mv <what you download something>
python.exe symspellpy_test.py
```


### Run vocab



```
cd data_preprocessing
python.exe Creating_Vocabulay_fixed.py
```

### Run train

```
cd model_training
python.exe XLSR_Wav2Vec2_jamo.py

```

### Run Client ( run by .py)
```
cd client
mkdir Assets
cp < ALL Assets and models that What step-(Run train)> 
python.exe main_run.py
```

## 6. pyinstaller (for exe file)
```
1. 아래 명령어를 입력한다. 아직 onefile로 하면 안되고, dist 폴더를 생성하게 하여, 부족한 부분을 채울 수 있도록 일단 1번 exe 파일을 생성한다.
pyinstaller --noconfirm --clean --distpath ./dist ^
			--add-data="symspell_jamo_dict.txt;." --add-data="Assets/vocab.json;Assets" --add-data="Assets/vocab_jamos.json;Assets"^
			--add-data="Assets/test_data.wav;Assets" --add-data="Assets/vocab_chars.json;Assets"^
			--hidden-import=pytorch --collect-data=torch --copy-metadata=torch ^
			--copy-metadata=tqdm ^
		    --hidden-import=tensorflow --copy-metadata=tensorflow --collect-data=tensorflow ^
			--hidden-import=transformers --copy-metadata=transformers --collect-data=transformers ^
			--copy-metadata=regex --copy-metadata=requests ^
			--copy-metadata=packaging --copy-metadata=filelock --copy-metadata=numpy ^
			--copy-metadata=tokenizers --copy-metadata=importlib_metadata ^
			--collect-data=librosa --copy-metadata=librosa ^
			--hidden-import="sklearn.utils._cython_blas" ^
			--hidden-import="sklearn.utils._typedefs" ^
			--hidden-import="sklearn.neighbors._partition_nodes" ^
			--hidden-import="scipy.special.cython_special" ^
			main_run.py




dist 폴더가 생성이 되고, dist\\main_run\\Assets에 모델 파일을 옮겨줘야 한다.
참고로 Assets에 모델은 미리 생성해야 한다. 

실행 파일은 아래와 같이 에러가 발생할 것이다.
RuntimeError: Failed to import transformers.models.wav2vec2.processing_wav2vec2 because of the following error (look up to see its traceback):
[Errno 2] No such file or directory: 'C:\\Users\\user_name\\AppData\\Local\\Temp\\_MEI109122\\transformers\\__init__.py'

그래서 transformers 폴더만 수작업으로 파일을 옮겨준다. 

수동으로 dist/main_run/transformers 폴더에 C:\Users\<users>\anaconda3\envs\STT\Lib\site-packages\transformers  파일을 모두 옮겨준다.
pyinstaller에서는 이슈로 인하여 transformers 폴더만 py 파일이 없다.
모두 C:\Users\<users>\anaconda3\envs\STT\Lib\site-packages 수동으로 옮겨주면 해결이 된다.

그 후 dist\\main_run\\main_run.exe을 실행하면 실행이 된다.
```



## 7. Reference
* Model/Code
   * facebook wav2vec-2 (https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)
* Dataset
   * AI Hub open domain dialog speech corpus data: 
   ** KSM [here] (https://aihub.or.kr/aidata/30707)
   ** KsponSpeech[here] (https://aihub.or.kr/aidata/105)
   ** KlecSpeech [here] (https://aihub.or.kr/aidata/30708)
* Client
   * pyqt5 swharden's Python-GUI-examples [hear] (https://github.com/swharden/Python-GUI-examples)
   * pyqt5 main (https://www.riverbankcomputing.com/static/Docs/PyQt5/)
   * pyinstaller main (https://pyinstaller.org/en/stable/) 

