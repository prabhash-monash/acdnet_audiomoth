import wavio;

import random;
import numpy as np;

import ffmpeg
import subprocess
import simpleaudio as sa;

#####################################################################################################








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


# wav_dir_input = "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/";
wav_dir_input = "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/esc50/ESC-50-master/audio/"
wav_dir_output = "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/output/";
# wav_filename_input = "manual_audio.wav";
wav_filename_input = "1-13571-A-46.wav";

wav_filename_output = "20Hz__1-13571-A-46.wav"

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

import keras
model_path = "/home/jana0009/acdnet_on_computer/acdnet/tf/resources/pretrained_models/acdnet20_20khz_fold4.h5"

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

import pandas as pd;
df = pd.read_csv('/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/esc50/ESC-50-master/meta/esc50.csv')
labels = list(df['category'].unique())
temp = df.iloc[df['category'].drop_duplicates().index][["target","category"]].to_dict('split')['data']
idx2class = {item[0]:item[1] for item in temp}

# [(None, 1, 30225, 1)]
# (1, 1, 30225, 1)
#(30226, 1)
# print('Testing - Val: Loss {:.2f}  Acc(top1) {:.2f}%'.format(val_loss, val_acc));
print(f"CLASSES = [{idx2class}]")

print(f"=============\n\nclass = {idx2class[int(predicted_class)]}")
