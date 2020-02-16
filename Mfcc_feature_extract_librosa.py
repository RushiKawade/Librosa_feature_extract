! git clone https://github.com/RushiKawade/Librosa_feature_extract.git
! ls



from glob import glob

# use local path in case of local run of program.
data_dir = './Librosa_feature_extract/voice_data/'
audio_files = glob(data_dir  + '*.mp3')


print('Number of data files loaded: '+ str(len(audio_files)))


import librosa
from librosa import feature
import numpy as np
import pandas as pd

# y = Time Series Array of audio files. Features will be extracted for each data point separately.
# sr is a sampling rate

speaker = []
res = []

for audio_path in audio_files:
  # ./pro1/normals/61-70968-0007 (online-audio-converter.com).mp3 is path of file to get name of person slice the file path as shown below. Numbers will be differant in your case.
  person = audio_path[37:]
  person = person[:-4]
  #print(person)
  speaker.append(person)


  y,sr = librosa.load(audio_path)
  features = librosa.feature.mfcc(y,sr)
  # You can use these features as it is or take mean for all 20 features for each time point by taking column-wise i.e. convert 20 * n array to 20 * 1by to get 1 feature vector per speaker
 # print(features.shape) # has a shape of 20 * n
  
  res.append(np.mean(features,axis=1)) # take row0wise mean


df = pd.DataFrame(res)
df['speaker']= speaker

print(df)