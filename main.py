import numpy as np
import load
import pca
import cv2
import os

path = "C:\\Users\\HWZ\\Github\\MADE2_Face_Recognize\\att_faces"

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
                elif filename == str(load.sampel_num) + ".pgm":
                    cnt = True
                elif cnt:
                    src = cv2.imread(path_str + '\\' + sub_path +
                                     '\\' + filename, cv2.IMREAD_GRAYSCALE)
                    src2 = cv2.resize(
                        src, (int(x_size), int(y_size)), cv2.INTER_AREA)
                    src2 = load.Centralization(src2.ravel())
                    man, dis = pca.detect(src2)
                    if man == path_str + '\\' + sub_path:
                        Correct_num += 1
                    # else:
                    #     print(sub_path)
                    #     print(man)
                    #     # cv2.imshow("src", src)
                    #     # cv2.imshow("det", pca.men_src[dis])
                    #     # cv2.waitKey(0)
                    test_num += 1
            cnt = False
    return Correct_num / test_num


def Full_test():
    result = []
    temp1 = []
    temp2 = []
    for i in range(2, 9):
        temp1 = [i]
        load.sampel_num = i
        for j in range(91, 98):
            temp2 = [j]
            pca.retention = i / 100
            load.training_load(path)
            temp2.append(pca.pca())
            temp2.append(test_all(path))
            print(i, ' ', j)
            pca.A = np.empty([x_size * y_size, 0])
            load.sample_mat = np.zeros([x_size * y_size, 0])
            load.men_list = []
            load.men_src = []
            load.col_num = 0
            load.src = 0

            pca.Cov_mat = np.empty([1, 1])
            pca.DR_mat = np.empty([1, 1])
            pca.Tr_mat = np.empty([1, 1])
            pca.DR_Num = np.empty([1, 1])
            pca.A = np.empty([x_size * y_size, 0])
            pca.Com_mat = np.empty([1, 1])
            pca.men_src = 0

        temp1.append(temp2)
    result.append(temp1)
    np.save("full_test_result", np.array(result, dtype=object))


if __name__ == "__main__":
    parameter = input("select mode: T,A,F or R?\n")
    if parameter == 'T':
        load.training_load(path)
        print("\nload over\n")
        pca.pca()
        print(load.col_num)
        print(np.shape(load.sample_mat))
    elif parameter == 'R':
        man, dis = pca.detect(load.load(path))
        print(man)
        print(dis)
    elif parameter == 'A':
        print(test_all(path))
    elif parameter == 'F':
        Full_test()
    else:
        print("inviable parameter.ヾ(•ω•`)o")
