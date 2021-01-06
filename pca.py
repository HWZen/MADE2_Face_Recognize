import numpy as np
import load
import cv2

x_size = load.x_size
y_size = load.y_size
retention = 0.91

Cov_mat = np.empty([1, 1])
DR_mat = np.empty([1, 1])
Tr_mat = np.empty([1, 1])
DR_Num = np.empty([1, 1])
A = np.empty([x_size * y_size, 0])
Com_mat = np.empty([1, 1])
men_src = 0


def pca():
    global x_size
    global y_size
    global retention
    global Cov_mat
    global DR_mat
    global DR_Num
    global A
    global Com_mat
    Cov_mat = np.dot(load.sample_mat, load.sample_mat.T) / (load.col_num - 1)
    np.save("Cov_mat", Cov_mat)
    print("Cov_mat: ", Cov_mat.shape)
    print("Cov_mat over.\n")

    DR_Num, DR_mat = np.linalg.eig(Cov_mat)

    DR_Num = DR_Num.astype(Cov_mat.dtype)
    DR_mat = DR_mat.astype(Cov_mat.dtype)
    np.save("DR_Num", DR_Num)
    np.save("DR_mat", DR_mat)

    # DR_Num = np.load("DR_Num.npy")
    # DR_mat = np.load("DR_mat.npy")

    print("DR_Num: ", len(DR_Num))
    print("DR_mat: ", np.shape(DR_mat))
    print("DR done.\n")

    Sum = 0
    for i in range(0, len(DR_Num)):
        Sum += DR_Num[i]

    cnt = 0
    for i in range(0, len(DR_Num)):
        # if cnt / Sum >= retention:
        if i >= 40:
            break
        else:
            A = np.insert(A, i, values=DR_mat[:, i], axis=1)
            cnt += DR_Num[i]
    A = A.T
    np.save("A", A)
    print("A: ", np.shape(A))
    print("A done.\n")

    Com_mat = np.dot(A, load.sample_mat)
    np.save("Com_mat", Com_mat)
    print("Com_mat: ", np.shape(Com_mat))
    print("Com_mat done.\n")

    print("PCA done.\n")


def detect(vec):
    global x_size
    global y_size
    global A
    global Com_mat
    global men_src
    A = np.load("A.npy")
    Com_mat = np.load("Com_mat.npy")
    men_list = np.load("men_list.npy")
    men_src = np.load("men_src.npy")
    DR_vec = np.dot(A, vec)

    mines = 0xffffff
    mines_id = 0

    for i in range(0, np.shape(Com_mat)[1]):
        if mines > np.linalg.norm(DR_vec - Com_mat[:, i]):
            mines = np.linalg.norm(DR_vec - Com_mat[:, i])
            mines_id = i
    # cv2.imshow("src", load.src)
    # cv2.imshow("det", men_src[mines_id])
    # cv2.waitKey(0)
    return men_list[mines_id], mines_id
