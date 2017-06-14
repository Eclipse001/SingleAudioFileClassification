# Single Audio File Classification

- Main objective: Distinguish the audio file which have normal quality of sound and the "broken" audio files (Contents of those audio files don't contain any meaning due to bad recording.) with machine learning methods.

# Tool and package used

- Python 2.7 : https://www.python.org/
- pyAudioAnalysis : https://github.com/tyiannak/pyAudioAnalysis
- ffmpeg

Use pip to install dependencies of pyAudioAnalysis:

- pip install numpy matplotlib scipy sklearn hmmlearn simplejson eyed3 pydub

# Pre-setup steps

- Makesure ffmpeg and the dependencies of pyAudioAnalysis are intalled.
- Download this repo.
- Download the pyAudioAnalysis repo from https://github.com/tyiannak/pyAudioAnalysis and replace the empty folder 'pyAudioAnaysis' inside this repo with the downloaded package.

- Put training sample audio files in the "Samples/Good" and "Samples/Bad" path (which are the relative path of the root path of this repo).


# Run the training script

- python train.py [midtermWindow] [midtermStep] [goodFolderPath] [badFolderPath] [tGoodFolderPath] [tBadFolderPath]

[midtermWindow] and [midtermStep] are the mid-term block size and step size used while training the classifier from the samples. (In seconds)

[goodFolderPath] and [badFolderPath] are the path of the folders contain the orginal training samples, there must be an ending slash in both of these arguments. For example: 'OrginalSamples/Good' will cause error but 'OrginalSamples/Good/' won't cause error.

[tGoodFolderPath] and [tBadFolderPath] are the path of the folders will contain the training samples that will actually used for training, that is, the copies of content inside [goodFolderPath] and [badFolderPath]. This is to prevent any modifcation of the orginal files while reducing the audio file's sampling rate. These two path must be exist before running the training script. And, files inside these two folders will be removed after training.

Th resulting svm model as well as its related files will be located in the folder named 'Models', the model file name will be 'svm'. 

- The new trained model will replace the previous one.
- The model type is SVM.
- If an 'unknown format" error message pops up during the training, this is probably due to sampling rate of a audio file is higher than 48K is not supported. In this case, try to uncomment the line 45 and 46 inside the train.py and redo the training.

# Run the processing script

- python script.py [modelPath] [filePath]

[modelPath] and [filePath] are the path to the trained model file and the path to the audio file to be processed.

- The output will be print to stdour, and it will be either 0 or 1, 0 stands for "Good" audio file, 1 stands for "Bad" audio file.
- If an 'unknown format" error message pops up during the process, this is probably due to sampling rate of a audio file is higher than 48K is not supported. In this case, try to uncomment all of the commented code and redo the process.
