import pyaudio
import wave
import webrtcvad
import numpy as np
import argparse

CHUNK = 320
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
p = pyaudio.PyAudio()
vad = webrtcvad.Vad()
vad.set_mode(1)

def main(filename):
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    audio_frame = []
    while True:
        try:
            data = stream.read(CHUNK)
            print(vad.is_speech(data, RATE))
            if vad.is_speech(data,RATE):
                audio_frame.append(data)
        except KeyboardInterrupt:
            break
    print('Saving as wav file...')
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    audio_data = b''.join(audio_frame)
    numpydata = np.frombuffer(audio_data, dtype=np.int16)
    print(numpydata.shape)
    wf.writeframes(audio_data)
    wf.close()

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='wav generator')
    parser.add_argument('filename', type=str, help='Output filename')
    #parser.add_argument('outdir', type=str, help='Output dir for image')
    args = parser.parse_args()
    print(args.filename)
    main(args.filename)
