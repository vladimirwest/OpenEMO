# OpenEMO
Team Pepe solution for Openvino hackathon: emotions and speaker recognition from voice

### Stage 0: Preapration
To use your own voice in speaker recognition model, please record it with following command: 
```
python3 voice_db/record_voice.py %filename%.wav
```
Recording should last about 10-15 seconds of continuous speech. Press ``Ctrl+C`` to interrupt the recording process.
After that you should generate a spectrograms for all audio files. That could be done with:
```
python3 create_base.py
```
It will convert all recorded voices to .npy files with spectrograms. 

### Stage 1: Demo
```
python3 demo.py
```
Note that speech sample should lasts about 10 seconds, otherwise it will not be recognized (due to speaker recognition model input_shape)
It will output list with all emotions probabilities and cosine distances for all speech samples which were stored in voice_db folder. In our demo we used a threshold (=0.35) to identificate a speaker.

### Used sources:
Speaker recognition: https://github.com/linhdvu14/vggvox-speaker-identification  
Emotions recognition: https://github.com/alexmuhr/Voice_Emotion
