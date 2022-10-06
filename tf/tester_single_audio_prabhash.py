# import simpleaudio as sa; from scipy import signal; sa.play_buffer( signal.resample((255*self.testX[1985,0,:,0]).astype(np.int16),int((30225*44100)/20000)),1,2,44100)

import pyaudio
import wave

audio_format = pyaudio.paInt16;
number_of_channels = 1;
sample_rate = 192000;
chunk_size = 4096
duration = 10 # seconds.


filename = 'test.wav'

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

print('Start recording...');

data = [];

total_samples = sample_rate * duration;
print(f"total_samples = {total_samples}");

while (total_samples > 0):
    samples = min(total_samples, chunk_size);
    data.append(stream.read(samples))
    total_samples -= samples;

print('Finished recording.');

# Stop the stream, close it, and terminate the pyaudio instance.

stream.stop_stream();
stream.close();
audio.terminate();

# Save the audio data as a WAV file.

wavefile = wave.open(filename, 'wb');
wavefile.setnchannels(number_of_channels);
wavefile.setsampwidth(audio.get_sample_size(audio_format));
wavefile.setframerate(sample_rate);
wavefile.writeframes(b''.join(data));
wavefile.close();



from tensorflow import  keras
import wavio

model_path = "/home/jana0009/acdnet_on_computer/acdnet/tf/resources/pretrained_models/acdnet20_20khz_fold4.h5"

model = keras.models.load_model(model_path);
sound = wavio.read(filename).data.T[0]
start = sound.nonzero()[0].min()
end = sound.nonzero()[0].max()
sound = sound[start: end + 1]  # Remove silent sections
# label = int(os.path.splitext(wav_file)[0].split('-')[-1])
# esc50_sounds.append(sound)

testX = data['x'];














