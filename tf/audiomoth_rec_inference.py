import wavio;

import random;
import numpy as np;

import ffmpeg
import subprocess
import simpleaudio as sa;

from tensorflow import keras;


#####################################################################################################

import pyaudio
import wave

import datetime

audio_format = pyaudio.paInt16;
number_of_channels = 1;
sample_rate = 192000;
chunk_size = 4096
duration = 5 # seconds.

rec_dir = '/home/pi/datasets/rec/'# '/home/jana0009/acdnet_on_computer/acdnet/rec_audiomoth/';
rec_timestamp_str = "{}".format(datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
rec_filename = 'rec_' + str(duration) + 'sec_' + rec_timestamp_str + '.wav'
rec_filepath = rec_dir + rec_filename;

# Create pyaudio instance and search for AudioMoth.

device_index = None;

audio = pyaudio.PyAudio()

print(f"audio.get_device_count() = {audio.get_device_count()}")

for i in range(audio.get_device_count()):
    print(f"{i} - {audio.get_device_info_by_index(i).get('name')}")

    if 'AudioMoth' in audio.get_device_info_by_index(i).get('name'):
        device_index = i;
        break;

if device_index == None:
    print('No AudioMoth found!');
    exit();


# Create pyaudio stream.
stream = audio.open(format=audio_format,
                    rate=sample_rate,
                    channels=number_of_channels,
                    input_device_index=device_index,
                    input=True,
                    frames_per_buffer=chunk_size);


# Append audio chunks to array until enough samples have been acquired.

print('Start recording... SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS');

data = [];

total_samples = sample_rate * duration;
print(f"total_samples = {total_samples}");

while (total_samples > 0):
    samples = min(total_samples, chunk_size);
    data.append(stream.read(samples))
    total_samples -= samples;

print('Finished recording. EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE');

# Stop the stream, close it, and terminate the pyaudio instance.

stream.stop_stream();
stream.close();
audio.terminate();

# Save the audio data as a WAV file.

wavefile = wave.open(rec_filepath, 'wb');
wavefile.setnchannels(number_of_channels);
wavefile.setsampwidth(audio.get_sample_size(audio_format));
wavefile.setframerate(sample_rate);
wavefile.writeframes(b''.join(data));
wavefile.close();





#####################################################################################################

random.seed(42)

def normalize(sound, factor):
    return sound / factor



def padding(sound, pad):
    return np.pad(sound, pad, 'constant')

def multi_crop(sound, input_length, n_crops):
    if(n_crops>1):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
    else:
        sounds = [sound];
    return np.array(sounds)



def preprocess(sound, inputLength, nCrops):
    sound = padding(sound, inputLength // 2);
    sound = normalize(sound, 32768.0),
    sound = multi_crop(sound, inputLength, nCrops);
    return sound;

index_filelist = 313;




# wav_dir_input = "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/";
wav_dir_input = rec_dir; # "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/esc50/ESC-50-master/audio/"
wav_dir_output = rec_dir + "output/"; #"/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/output/";
# wav_filename_input = "manual_audio.wav";
wav_filename_input = rec_filename; # "1-13571-A-46.wav";

wav_filename_output = "20Hz__" + rec_filename;


use_directory_list = False;

if(use_directory_list):
    filelist_textfile = "/home/pi/datasets/esc50/ESC-50-master/filelist.txt"
    # filelist_data = open(filelist_textfile, "r");
    # filelist_as_list = filelist_data.split("\n");
    filelist_as_list = [];
    with open(filelist_textfile) as textfile:
        filelist_as_list = textfile.readlines();
        filelist_as_list = [line.rstrip() for line in filelist_as_list];

    wav_filename_input = filelist_as_list[index_filelist];#  "audio_1-100210-A-36.wav";
    print(f"working on [{wav_filename_input}]");

   
    # wav_dir_input = "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/";
    # wav_dir_input = "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/esc50/ESC-50-master/audio/"
    wav_dir_input = "/home/pi/datasets/esc50/ESC-50-master/audio/";
    wav_dir_output = "/home/pi/datasets/esc50/output/"; 
    # wav_filename_input = "manual_audio.wav";

    wav_filename_output = "20Hz__" + wav_filename_input;

    # model_path = "/home/jana0009/acdnet_on_computer/acdnet/tf/resources/pretrained_models/acdnet20_20khz_fold4.h5"

model_path = "/home/pi/ACDNET/acdnet/tf/resources/pretrained_models/acdnet20_20khz_fold4.h5";


wav_file_fullpath_input = wav_dir_input + wav_filename_input;
wav_file_fullpath_output = wav_dir_output + wav_filename_output;
wav_file_fullpath_tmp_play = wav_dir_output + "temp_" + wav_filename_output;

wavio_data = wavio.read(wav_file_fullpath_input);

sr_original = wavio_data.rate;  # e.g. 192000 for audiomoth.
sr_output = 20000;
sr_tmp_play = 22050;


ffmpeg_command = 'ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(wav_file_fullpath_input,
                                                                         sr_output,
                                                                         wav_file_fullpath_output);
print(f"FFMPEG COMMAND: [{ffmpeg_command}]");
subprocess.call(ffmpeg_command, shell=True);
print("Converting : ")
print(wav_file_fullpath_input);
print("--> OUTPUT: ")
print(wav_file_fullpath_output);


wavio_data_converted = wavio.read(wav_file_fullpath_output);
print(wavio_data)
print(wavio_data_converted)


inputLength = 30225;
print(f"inputLength = {inputLength}");

duration = 5;   # 4 seconds;
sound_original = wavio_data.data
sound_converted = wavio_data_converted.data;




start = 100000 // 2;# sound.nonzero()[0].min()
end =  start + inputLength; # int(sr_original * duration - 1); # sound.nonzero()[0].max()
print(f"sound_original =>  length = {sound_original.shape[0]}")
print(f"sound_converted => length = {sound_converted.shape[0]}")

sound_cropped = sound_converted[start: end]


_nCrops = 1;

# wav_file_fullpath_temp_play =

ffmpeg_cmd_to_play20000Hz = 'ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(wav_file_fullpath_output,
                                                                         sr_tmp_play,
                                                                         wav_file_fullpath_tmp_play);
print(f"TEMPORILY CONVERTING FOR PLAYING:: FFMPEG COMMAND: [{ffmpeg_command}]");
subprocess.call(ffmpeg_cmd_to_play20000Hz, shell=True);
wavio_data_tmp_play = wavio.read(wav_file_fullpath_tmp_play);
sound_tmp_play = wavio_data_tmp_play.data.T;


# sa.play_buffer(sound, 1, 2, sr_original)
_handle_sa = sa.play_buffer(sound_cropped, 1, 2, sr_tmp_play)
# preprocess_funcs = preprocess_setup(inputLength=_inputLength_in_seconds, nCrops=_nCrops);

_handle_sa.wait_done();
print("done")
# sound_preprocessed = preprocess(sound_converted, inputLength=_inputLength_in_seconds, nCrops=_nCrops);


model = keras.models.load_model(model_path);
print('Model Loaded.');
model.summary();
print(f"sound_cropped = {sound_cropped.shape}")
# x = sound_preprocessed
x = np.asarray(sound_cropped).astype(np.float32)

# tf.expand_dims(image, axis=0).shape.as_list()
x_ = keras.backend.expand_dims(x,axis=0)
x_ = keras.backend.expand_dims(x_,axis=0)
print(x_.shape)
scores = model.predict(x_, batch_size=len(x), verbose=0);
predicted_class = scores.argmax(-1);
print(f"predicted_class = {predicted_class}")

# import pandas as pd;
# df = pd.read_csv('/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/esc50/ESC-50-master/meta/esc50.csv')
# labels = list(df['category'].unique())
# temp = df.iloc[df['category'].drop_duplicates().index][["target","category"]].to_dict('split')['data']
# idx2class = {item[0]:item[1] for item in temp}
# 
# # [(None, 1, 30225, 1)]
# # (1, 1, 30225, 1)
# #(30226, 1)
# # print('Testing - Val: Loss {:.2f}  Acc(top1) {:.2f}%'.format(val_loss, val_acc));
# print(f"CLASSES = [{idx2class}]")
# 
# print(f"=============\n\nclass = {idx2class[int(predicted_class)]}")
