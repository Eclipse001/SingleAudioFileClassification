from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from os import path
from shutil import copyfile

import os
import sys
import time
import wave

mtw = float(sys.argv[1])                # Command line argument 1 is the mid-term window.
mts = float(sys.argv[2])                # Command line argument 2 is the mid-term step.
oClassADirectory = sys.argv[3]          # Command line argument 3 is the folder containing class A samples.
oClassBDirectory = sys.argv[4]          # Command line argument 4 is the folder containing class B samples.

classADirectory = oClassADirectory + '-processed'
classBDirectory = oClassBDirectory + '-processed'

def removeDirectory(dirPath):
    for fileName in os.listdir(dirPath):
        os.remove(dirPath + '/' +fileName)


def preprocess(orginalDirectory, resDirectory):
    
    for fileName in os.listdir(orDirPath):
        fileFullPath = orginalDirectory + '/' + fileName
        resFileName = str(time.time())+'.wav'
        
        if checkFileProp(orFullPath):
            print >> sys.stderr, 'Sampling rate too large, changing to 48K while copying to sample folder: ' + rFileName +'...'
            os.system('ffmpeg -i ' + fileFullPath + ' -ar 48000 ' + resDirectory + '/' + resFileName)
        else:
            print >> sys.stderr, 'Copying to the tmp folder as ' + resFileName + '...'
            copyfile(fileFullPath, resDirectory + '/' + resFileName)

def checkFileProp(filePath):
    FLAG_CONVERT_SR = False
    
    print >> sys.stderr, 'Checking training set file: ' + filePath.split("/")[-1] + '...'
    
    try:
        sr = wave.openfp(filePath, 'r').getframerate()
        if sr > 48000:
            FLAG_CONVERT_SR = True
    except:
        FLAG_CONVERT_SR =True
    
    return FLAG_CONVERT_SR


preprocess(oClassADirectory, classADirectory)
preprocess(oclassBDirectory, classBDirectory)

aT.featureAndTrain([goodDirPath, badDirPath], mtw, mts, aT.shortTermWindow, aT.shortTermStep, 'svm', "Models/svm")

removeDirectory(classADirectory)
removeDirectory(classBDirectory)
