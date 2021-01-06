import numpy as np
import load
import pca
import cv2
import os

path = "E:\\Github\\MADE2_Face_Recognize\\att_faces\\s12\\9.pgm"

x_size = load.x_size
y_size = load.y_size


def test_all(path_str):
    global x_size
    global y_size
    cnt = False
    test_num = 0
    Correct_num = 0
    for sub_path in os.listdir(path_str):
        if not os.path.isdir(path_str + '\\' + sub_path):
            continue
        else:
            for filename in os.listdir(path_str + '\\' + sub_path):
                if not os.path.isfile(path_str + '\\' + sub_path + '\\' + filename):
                    continue
                elif filename == "6.pgm":
                    cnt = True
                elif cnt:
                    src = cv2.imread(path_str + '\\' + sub_path + '\\' + filename, cv2.IMREAD_GRAYSCALE)
                    src2 = cv2.resize(src, (int(x_size), int(y_size)), cv2.INTER_AREA)
                    src2 = load.Centralization(src2.ravel())
                    man, dis = pca.detect(src2)
                    if man == path_str + '\\' + sub_path:
                        Correct_num += 1
                    else:
                        print(sub_path)
                        print(man)
                        # cv2.imshow("src", src)
                        # cv2.imshow("det", pca.men_src[dis])
                        # cv2.waitKey(0)
                    test_num += 1
            cnt = False
    return Correct_num / test_num


if __name__ == "__main__":
    parameter = input("select mode: T,A or R?\n")
    if parameter == 'T':
        load.training_load("E:\\Github\\MADE2_Face_Recognize\\att_faces")
        print("\nload over\n")
        pca.pca()
        print(load.col_num)
        print(np.shape(load.sample_mat))
    elif parameter == 'R':
        man, dis = pca.detect(load.load(path))
        print(man)
        print(dis)
    elif parameter == 'A':
        print(test_all("E:\\Github\\MADE2_Face_Recognize\\att_faces"))
    else:
        print("inviable parameter.ヾ(•ω•`)o")
