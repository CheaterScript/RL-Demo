import cupy as cp
import numpy as np
import h5py

def load_dataset(use_cupy=True):
    array_module = cp if use_cupy else np
    
    train_dataset = h5py.File('data/train_catvnoncat.h5', "r")
    train_set_x_orig = array_module.array(train_dataset["train_set_x"][:])
    train_set_y_orig = array_module.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('data/test_catvnoncat.h5', "r")
    test_set_x_orig = array_module.array(test_dataset["test_set_x"][:])
    test_set_y_orig = array_module.array(test_dataset["test_set_y"][:])

    # 字符串数据总是使用NumPy
    classes = np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes