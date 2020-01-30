import pandas as pd
import sys
import argparse
import os
import cv2 as cv
import time
import numpy as np
import librosa
import pyaudio
import webrtcvad
from threading import Thread
import queue
#from aiy.leds import Leds, Color
#from aiy.leds import RgbLeds
import glob, os
from scipy.signal import lfilter, butter
import sigproc
import constants as c
from scipy.spatial.distance import cdist, euclidean, cosine
import time
#import json
#import asyncio
#from stream import connection

#conn = connection.stream()

def load_wav(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()
	return audio


def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


def remove_dc_and_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1,-1], [1,-alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


def get_fft_spectrum(signal):
	signal *= 2**15

	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE)
	signal = sigproc.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = sigproc.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT))
	fft_norm = normalize_frames(fft.T)
	rsize = 500 
	rstart = int((fft_norm.shape[1]-rsize)/2)
	out = fft_norm[:,rstart:rstart+rsize]

	return out


class Config:
    def __init__(self, n_mfcc = 26, n_feat = 13, n_fft = 552, sr = 22050, window = 0.4, test_shift = 0.1):
        self.n_mfcc = n_mfcc
        self.n_feat = n_feat
        self.n_fft = n_fft
        self.sr = sr
        self.window = window
        self.step = int(sr * window)
        self.test_shift = test_shift
        self.shift = int(sr * test_shift)
config = Config()

srNet = cv.dnn.readNet('speaker_recognition_model.bin',
		       'speaker_recognition_model.xml')
#srNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

df = pd.DataFrame(columns = ['speaker', 'spec', 'embedding'])
for file in glob.glob("voice_db/*.npy"):
    spec = np.load(file)
    spec = spec.astype('float32')
    spec_reshaped = spec.reshape(1, 1, spec.shape[0], spec.shape[1])
    srNet.setInput(spec_reshaped)
    pred = srNet.forward()
    emb = np.squeeze(pred)
    #print(file[9:-4])
    row = [file[9:-4], spec, emb]
    df.loc[len(df)] = row

print(df)

emotionsNet = cv.dnn.readNet('emotions_model.bin',
                              'emotions_model.xml')
#emotionsNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

process = True
#
# Microphone frames capture thread
#
sr = 48000

def to_queue(frames, timestamp_start):
    timestamp_end = time.time()
    #print(timestamp_end)
    print('put to queue')
    #audio_data = []
    #for frame in frames:
    #    audio_data.append(b''.join(frame))
    d = {}
    d['audio'] = np.frombuffer(b''.join(frames), dtype=np.int16)
    #d['timestamp_start'] = timestamp_start
    #d['timestamp_end'] = timestamp_end
    return d

#async def sendPermanentData(predictors):
#    await conn.send_audio(predictors)


framesQueue = queue.Queue()
def framesThreadBody():
    CHUNK = 960
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000

    p = pyaudio.PyAudio()
    vad = webrtcvad.Vad()
    vad.set_mode(2)
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    print("* recording")
    false_counter = 0
    audio_frame = []
    #global framesQueue, process
    timestamp_start = time.time()
    while process:
        #print(framesQueue.qsize())
        data = stream.read(CHUNK)
        #print(vad.is_speech(data, RATE))
        #await sendPermanentData(vad.is_speech(data, RATE))

        if not vad.is_speech(data, RATE):
            false_counter += 1
            if false_counter >= 30:
                #print('MAKING NEW AUDIO FRAME')
                #do smt with previous audio frame
                if len(audio_frame) > 250:
                    framesQueue.put(to_queue(audio_frame,timestamp_start))
                timestamp_start = time.time()
                audio_frame = []
                false_counter = 0

        if vad.is_speech(data, RATE):
            false_counter = 0
            audio_frame.append(data)
            if len(audio_frame) > 300:
                print(timestamp_start)
                framesQueue.put(to_queue(audio_frame,timestamp_start))
                timestamp_start = time.time()
                #print('MAKING NEW AUDIO FRAME')
                audio_frame = []


#led_dict = {'neutral': (255, 255, 255), 'happy': (0, 255, 0), 'sad': (0, 255, 255), 'angry': (255, 0, 0), 'fearful': (0, 0, 0), 'disgusted':  (255, 0, 255), 'surprised':  (255, 255, 0)} 
#leds = Leds()


# Skip the first frame to wait for camera readiness
#framesQueue.get()
def iterate():
    while process:
        frame = None
        while frame is None:
            try:
                frame = framesQueue.get_nowait()
            except queue.Empty:
                frame = None

        wav = frame.get('audio')
        wav = np.array(wav, dtype='float32')
        _min, _max = float('inf'), -float('inf')
        local_results = []
        X = []
        classes = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
        # Iterate over sliding 0.4s windows of the audio file
        for i in range(int((wav.shape[0]/sr-config.window)/config.test_shift)):
            X_sample = wav[i*config.shift: i*config.shift + config.step] # slice out 0.4s window
            X_mfccs = librosa.feature.mfcc(X_sample, sr, n_mfcc = config.n_mfcc, n_fft = config.n_fft, hop_length = config.n_fft)[1:config.n_feat + 1] # generate mfccs from sample 
            _min = min(np.amin(X_mfccs), _min)
            _max = max(np.amax(X_mfccs), _max) # check min and max values
            X.append(X_mfccs) # add features of window to X

    # Put window data into array, scale, then reshape
        X = np.array(X, dtype = 'float32')
        X = (X - _min) / (_max - _min)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    # Feed data for each window into model for prediction
        for i in range(X.shape[0]):
            window = X[i].reshape(1, 1, X.shape[2], X.shape[1])
            emotionsNet.setInput(window)
    #    local_results.append(model.predict(window))
            local_results.append(emotionsNet.forward())

    # Aggregate predictions for file into one then append to all_results
        local_results = (np.sum(np.array(local_results), axis = 0)/len(local_results))[0]
        local_results = list(local_results)
        print(local_results)
        prediction = np.argmax(local_results)
        print(classes[prediction])
        #print(led_dict.get(classes[prediction]))
        #leds.update(Leds.rgb_on(led_dict.get(classes[prediction])))
        #print(wav.shape)
        resampled_wav = librosa.resample(wav,48000,16000)
        spec = get_fft_spectrum(resampled_wav)
        enroll_embs = np.array([emb.tolist() for emb in df['embedding']])
        spec = spec.astype('float32')
        spec_reshaped = spec.reshape(1, 1, spec.shape[0], spec.shape[1])
        srNet.setInput(spec_reshaped)
        pred = srNet.forward()
        emb = np.squeeze(pred)
        emb = emb.reshape(1, emb.shape[0])
        dist_list = cdist(emb, enroll_embs, metric=c.COST_METRIC)
        distances = pd.DataFrame(dist_list, columns = df.speaker)
        print(distances)
        print('_________________________________')
        #del frame['audio']
        #frame['emotion_clases'] = classes
        #frame['emotion_probability'] = np.squeeze(np.array(local_results)).tolist() #['{:.3f}'.format(x) for x in local_results]
        #frame['speaker_arr'] = list(df.speaker.values)
        #frame['speaker_probability'] = np.squeeze(dist_list).tolist() #['{:.3f}'.format(x) for x in dist_list]
        #await sendPermanentData(frame)




#def nets_iterate():
#    loop = asyncio.new_event_loop()
#    asyncio.set_event_loop(loop)
#    asyncio.ensure_future(iterate())
#    loop.run_forever()

#def nets_frames():
#    loop = asyncio.new_event_loop()
#    asyncio.set_event_loop(loop)
#    asyncio.ensure_future(framesThreadBody())
#    loop.run_forever()

framesThread = Thread(target=framesThreadBody)
iterateThread = Thread(target = iterate)
framesThread.start()
iterateThread.start()
framesThread.join()
iterateThread.join()
