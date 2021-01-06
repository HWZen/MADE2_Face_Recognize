import cv2
import numpy as np

import os

x_size = 92
y_size = 112

sample_mat = np.zeros([x_size * y_size, 0])
men_list = []
men_src = []
col_num = 0
src = 0


def training_load(path_str):
    global x_size
    global y_size
    global sample_mat
    global col_num
    global men_list
    global men_src
    for sub_dir in os.listdir(path_str):
        if os.path.isfile(path_str + '\\' + sub_dir):
            continue
        else:
            print("loading " + sub_dir)
            load_file(path_str + '\\' + sub_dir)
    np.save("men_list", men_list)
    np.save("men_src", men_src)


def load_file(path_str):
    global x_size
    global y_size
    global sample_mat
    global col_num
    global men_list
    global men_src
    cnt = 0
    temp = np.zeros([x_size * y_size, ])
    for filename in os.listdir(path_str):
        if os.path.isdir(path_str + '\\' + filename):
            continue
        elif filename == "6.pgm":
            break
        else:
            img = cv2.imread(path_str + '\\' + filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (int(x_size), int(y_size)), cv2.INTER_AREA)
            men_src.append(img)
            img = img.ravel()
            temp += img
            cnt += 1
    men_list += [path_str]
    temp /= cnt
    sample_mat = np.insert(sample_mat, col_num, values=Centralization(temp), axis=1)
    col_num += 1


def Centralization(Vector):
    global x_size
    global y_size
    Sum = 0
    cnt = 0
    Vector = Vector.astype(np.float)
    for i in range(0, len(Vector)):
        Sum += Vector[i]
        cnt += 1
    Sum /= cnt
    for i in range(0, len(Vector)):
        Vector[i] -= Sum
    return Vector


def load(path_str):
    global x_size
    global y_size
    global src
    if not os.path.isfile(path_str):
        print("can't find file.")
        exit()
    else:
        src = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        _src = cv2.resize(src, (int(x_size), int(y_size)), cv2.INTER_AREA)
        _src = Centralization(_src.ravel())
        return _src
