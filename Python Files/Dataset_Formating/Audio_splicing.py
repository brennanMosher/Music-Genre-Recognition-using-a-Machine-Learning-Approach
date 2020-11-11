from pydub import AudioSegment
import os
import math
from pathlib import Path

'''
Splice wav files into multiple segments.
'''

LENGTH = 3 # Set splice length in seconds


def splice(audioPath):
    try:
        os.mkdir('splice') # Need to figure out where to put this
    except OSError:
        print("Creation of the directory failed")

    audio = AudioSegment.from_wav(audioPath)
    count = math.ceil(audio.duration_seconds/LENGTH) # Do we want the last part of audio?
    t1 = 0
    t2 = LENGTH*1000

    for i in range(count):
        newAudio = audio[t1:t2]
        newPath = 'splice/'+Path(audioPath).stem+'_splice'+str(i)+'.wav'
        newAudio.export(newPath, format="wav")
        t1 = t2
        t2 = t2 + LENGTH*1000

