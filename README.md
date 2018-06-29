# Single Audio File Classification

- Main objective: Distinguish the audio file which have normal quality of sound and the "broken" audio files (Contents of those audio files don't contain any meaning due to bad recording.) with machine learning methods.

## Tool and package used

- Python 2.7
- pyAudioAnalysis : https://github.com/tyiannak/pyAudioAnalysis
- ffmpeg

- Python 3.6 and keras for Conv1d model.

Use pip to install dependencies of pyAudioAnalysis:

- pip install numpy matplotlib scipy sklearn hmmlearn simplejson eyed3 pydub

## Pre-setup steps

- Makesure ffmpeg and the dependencies of pyAudioAnalysis are intalled.
- Download this repo.


## Run the training script

- python train.py [midtermWindow] [midtermStep] [classAFolder] [classBFolder]

[midtermWindow] and [midtermStep] are the mid-term block size and step size used while training the classifier from the samples. (In seconds)

Th resulting svm model as well as its related files will be located in the folder named 'Models', the model file name will be 'svm'. 

- The new trained model will replace the previous one.
- The model type is SVM.

## Run the processing script

- python test.py [modelPath] [audioFilePath]

- The output will be print to stdout, and it will be either 0 or 1.
- If an 'unknown format" error message pops up during the process, this is probably due to sampling rate of a audio file is higher than 48K is not supported. In this case, try to uncomment all of the commented code and redo the process.

## Conv1d model:

- python3 conv1d.py [classAFolder] [classBFolder]

- ### This model is not fully implemented for now, directly feeding raw signal of .wav files without any preprocessing may cause the system run out of memory. Apply zero padding to the audio files which has shorter length may not be a good idea as well.

