from pyAudioAnalysis import audioTrainTest as aT
from os import path

import os
import sys

modelPath = sys.argv[1]         # Command line argument 1 is the path of the final model.
filePath = sys.argv[2]          # Command line argument 2 is the path of the audio file.


fileName = filePath.split("/")[-1]

#===============================================================
'''
FLAG_CONVERT_SR = False

print >> sys.stderr, 'Processing ' + fileName + '...'

try:
    sr = wave.openfp(filePath, 'r').getframerate()
    if sr > 48000:
        FLAG_CONVERT_SR = True
except:
    FLAG_CONVERT_SR =True

if FLAG_CONVERT_SR:
    print >> sys.stderr, "Sampling rate too large, converting to 48K as tmp.wav..."
    os.system('ffmpeg -i ' + filePath + ' -ar 48000 tmp.wav')
    filePath = 'tmp.wav'
'''
#===============================================================

res = aT.fileClassification(filePath, modelPath, 'svm')

#===============================================================
'''
if FLAG_CONVERT_SR:
    print >> sys.stderr, "Removing tmp.wav..."
    os.remove('tmp.wav')
'''   
#===============================================================

if rawRes == 1:
    res = "Bad"

print >> sys.stderr, "Result for file: " + fileName + " is " + str(res) + '.'

