import numpy as np;

import matplotlib.pyplot as plt

dataset_file_path_20kHz = "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/esc50/test_data_20khz/fold1_test4000.npz";
dataset_file_path_44kHz = "/home/jana0009/acdnet_on_computer/acdnet/tf/datasets/esc50/test_data_44khz/fold1_test4000.npz"

dataset_20kHz = np.load(dataset_file_path_20kHz, allow_pickle=True);
dataset_44kHz = np.load(dataset_file_path_44kHz, allow_pickle=True);

print("20kHz DATASET:")
print(dataset_20kHz.files)
print(dataset_20kHz['x'].shape)

print("\n44kHz DATASET:")
print(dataset_44kHz.files);
print(dataset_44kHz['x'].shape)


data_index = 1;
datapoint_20kHz = dataset_20kHz['x'][data_index,0,:,0];
plt.plot(datapoint_20kHz);
datapoint_44kHz = dataset_44kHz['x'][data_index,0,:,0];
plt.plot(datapoint_44kHz);
plt.show()
