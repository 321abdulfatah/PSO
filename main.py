#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plotter
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import simpleaudio as sa
import os
import soundfile as sf
import noisereduce as nr

import PSO
import plotting_functions as plotting

curr_path = os.path.dirname(os.path.realpath(__file__))

# audio file names:

file_x1=os.path.join(curr_path,'Male Voice.wav')
file_x2=os.path.join(curr_path,'Female Voice.wav')
mixed=os.path.join(curr_path,"mix.wav")
file_s1 = os.path.join(curr_path,"1st voice.wav")
file_s2 = os.path.join(curr_path,"2nd voice.wav")
file_s1_r = os.path.join(curr_path,"1st voice reduced noise.wav")
file_s2_r = os.path.join(curr_path,"2nd voice reduced noise.wav")
# plot titles:
title_x = "Observed Data x"
title_s = "Estimated Source s"
title_X = "Observed Data X"
title_S = "Estimated Sources S"

# """
# play audio files:
print("Playing", file_x1, "...")
wave_obj = sa.WaveObject.from_wave_file(file_x1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_x2, "...")
wave_obj = sa.WaveObject.from_wave_file(file_x2)
play_obj = wave_obj.play()
play_obj.wait_done()
# """

# load audio files:
sample_freq_1, x1 = read(file_x1)
sample_freq_2, x2 = read(file_x2)

le=int(min(len(x1),len(x2)))

S=np.zeros((2,le))
S[0,:]=x1[0:le]
S[1,:]=x2[0:le]
A= np.array([[4.5359,-5.6223],[-6.0723, -9.1858]])

X=np.dot(A,S) 

write(mixed,sample_freq_1,X[0].astype(np.int16))

# play mixed audio files:
print("Playing", mixed, "...")
wave_obj = sa.WaveObject.from_wave_file(mixed)
play_obj = wave_obj.play()
play_obj.wait_done()

# """

# plot raw audio signals:
plotting.plot_signals(X, sample_freq_1, title_x)
# create a scatter plot of raw audio signals:
plotting.scatter_plot_signals(X, title_X, 'x')


# --------------------PSO ALGORITHM--------------------


# run PSO algorithm:
R = PSO.pso(X,le)

sf.write(file_s1,R[0], sample_freq_1) #test one voice   
sf.write(file_s2,R[1], sample_freq_2) 

#reduce noise to delete  low voice PSO didn't remove
reduced_noise_f = nr.reduce_noise(y=R[0], sr=sample_freq_1)
reduced_noise_m = nr.reduce_noise(y=R[1], sr=sample_freq_2)

sf.write(file_s1_r,reduced_noise_f, sample_freq_1) #test one voice   
sf.write(file_s2_r,reduced_noise_m, sample_freq_2) 

# --------------------PLAYING RESULTS--------------------

# convert source numpy arrays to .WAV files:

# play audio files:
print("Playing", file_s1, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_s2, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s2)
play_obj = wave_obj.play()
play_obj.wait_done()


print("Playing", file_s1_r, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s1)
play_obj = wave_obj.play()
play_obj.wait_done()
print("Playing", file_s2_r, "...")
wave_obj = sa.WaveObject.from_wave_file(file_s2)
play_obj = wave_obj.play()
play_obj.wait_done()



print("")



# display plots:
plotter.show()


print("\n\nDone!\n")