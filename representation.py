import numpy as np
# import matplotlib.pyplot as plt
import copy
from scipy.spatial.distance import pdist


def uniformity(R, m='euclidean'):
    return min(pdist(R, m))


def dist(p, q):  # 2-norm distance
    res = 0
    for i in range(len(p)):
        res += (p[i] - q[i]) ** 2

    return float(np.sqrt(res))


def URzero(B, k):
    """

    :param B: skyline set.
    :param k: number of representative points.
    :return: tuple. (uniformity value, <list> index of representative points in B)
    """

    n = len(B)
    T_val = np.zeros((k, n))
    T_idx = [[0] * n for _ in range(k)]

    # T[0][i] initialized to inf.
    for i in range(0, n):
        T_val[0][i] = np.inf
        T_idx[0][i] = [i]

    #
    for i in range(1, k):
        for j in range(n - i):

            temp_val = []
            temp_idx = []
            for ll in range(j + 1, n - i + 1):
                idx = T_idx[i - 1][ll][:]
                val = T_val[i - 1][ll]
                d = dist(B[j], B[ll])
                if d <= val:
                    temp_val.append(d)
                else:
                    temp_val.append(val)
                temp_idx.append(idx + [j])

            max_idx = int(np.argmax(temp_val))
            max_val = np.max(temp_val)

            T_val[i][j] = max_val
            T_idx[i][j] = temp_idx[max_idx]

    argmax = int(np.argmax(T_val[k - 1]))
    return np.max(T_val[k - 1]), T_idx[k - 1][argmax]


def UR(B, k):  # bad implementation!
    """
    :param B:  数据第一行id=0 为了补全索引的位置，不参与计算，所以索引是几，就是第几个
    :param k:
    :return:
    """
    n = len(B) - 1
    T = [[0] * (n + 1) for _ in range(k + 1)]
    for i in range(1, n + 1):
        T[1][i] = {float('inf'): [i]}

    #      0  1  2  3  4  5
    # B = [x, -, -, -, -, -]
    for i in range(2, k + 1):  # 2 3 4
        for j in range(1, n + 2 - i):  # 1 2 3 4 5

            res = {}
            for l in range(j + 1, n - i + 3):  # 2 3 4 5    3 4     4
                last = float(list(T[i - 1][l].keys())[0])  # T[i-1][l]

                idx = T[i - 1][l][last][:]
                d = dist(B[j], B[l])
                if d <= last:
                    res[d] = [j] + idx
                else:
                    res[last] = [j] + idx

            # if res:
            T[i][j] = {max(res.keys()): res[max(res.keys())]}

    temp = 0
    for each in T[k]:
        if isinstance(each, dict):
            if list(each.keys())[0] > temp:
                temp = list(each.keys())[0]
                idx = list(each.values())[0]

    return temp, idx


# TODO recursive method not work
def UR_rec(B, k):
    n = len(B)

    def T(i, j):
        if i == 1:
            return float('inf')
        else:
            res = []
            for ll in range(j + 1, n - i + 3):
                # d = np.array([B[j], B[ll]])
                # min_val = min((float(pdist(d)), T(i - 1, j)))
                min_val = min((dist(B[j], B[ll]), T(i - 1, j)))
                # d = dist(B[j], B[ll])
                # cpr = (d, T(i - 1, j))
                # min_val = min(cpr)

                res.append(min_val)

            print('res:', res)
            val = max(res)
            return val

    opt_uni = []
    for s in range(0, n - k + 1):
        t = T(k, s)
        print(t)
        opt_list = []
        opt_uni.append(t)

    return max(opt_uni)


def CR(B, k):
    """
    :param B:   数据第一行id=0 与索引同步
    :param k:
    :return:
    """

    def delta(j_, l_):
        resmin = []
        for m in range(j_, l_ + 1):
            resmin.append(min(dist(B[j_], B[m]), dist(B[m], B[l_])))
        return max(resmin)

    # B = isndarray(B)
    n = len(B) - 1  # 5
    T = [[0] * (n + 1) for _ in range(k + 1)]  # 这个矩阵是补了0的，位置从1开始
    TT = [[0] * (n + 1) for _ in range(k + 1)]
    for j in range(1, n + 1):
        T[1][j] = {dist(B[j], B[n]): [j]}
    # print("first: ", T[1])
    for i in range(2, k + 1):  # 2 3
        for j in range(1, n - i + 1):  # 1 2 3      1 2
            llist = {}
            for l in range(j + 1, n - i + 2):

                last = max(T[i - 1][l].keys())
                id = copy.deepcopy(T[i - 1][l][last])  # id 就是l上一层

                # llist.append(max(delta(j, l), T[i - 1][l]))
                de = delta(j, l)
                if de > last:
                    llist[de] = id + [j]  # 这个id里面存了l的信息，因此只需要加l就好了

                else:
                    llist[last] = id + [j]

            # if llist:
            T[i][j] = {min(llist.keys()): llist[min(llist.keys())]}

    # print("second: ", T[2])
    for k in range(1, k + 1):
        for j in range(1, n + 1):
            if isinstance(T[k][j], dict):
                last_T = max(T[k][j].keys())
                id_T = copy.deepcopy(T[k][j][last_T])
            else:
                continue
            if dist(B[1], B[j]) > last_T:
                TT[k][j] = {dist(B[1], B[j]): id_T}
            else:
                TT[k][j] = {last_T: id_T}
    ############################ 第一位补了0， 后面出来的结果从1开始 ###################################

    TT[k].remove(0)
    # print("k-order: ", TT[k])
    minest = float('inf')
    idx = []
    for each in TT[k]:
        if isinstance(each, dict):
            if list(each.keys())[0] < minest:
                minest = list(each.keys())[0]
                idx = list(each.values())[0]

    return idx


def e(er, eb):  # er eb 是一个点
    if isinstance(er, float) and isinstance(eb, float):
        return eb / er
    else:
        res = []
        dim = len(er)
        for i in range(dim):
            res.append(eb[i] / er[i])
        return max(res)


def ER(B, k):
    def delta(j_, l_):
        resmin = []
        for m in range(j_, l_ + 1):
            resmin.append(min(e(B[j_], B[m]), e(B[l_], B[m])))
        return max(resmin)

    # B = isndarray(B)
    n = len(B) - 1  # 8
    T = [[0] * (n + 1) for i in range(k + 1)]
    # 初始化第一层
    for j in range(1, n + 1):
        T[1][j] = {e(B[j], B[n]): [j]}
    # print(T[1])
    # 第二层的动态规划
    for i in range(2, k + 1):
        for j in range(i, n + 1):
            llist = {}
            for l in range(j + 1, n - i + 2 + 1):
                # llist.append(max(delta(j, l), T[i - 1][l]))
                last = list(T[i - 1][l].keys())[0]
                idx = copy.deepcopy(T[i - 1][l][last])  # id 就是l上一层

                if delta(j, l) >= last:
                    llist[delta(j, l)] = idx + [j]  # 这个id里面存了l的信息，因此只需要加l就好了
                else:
                    llist[last] = idx + [j]

            if llist:
                T[i][j] = {min(llist.keys()): llist[min(llist.keys())]}

    res = {}
    for j in range(1, n - k + 1 + 1):

        if isinstance(T[k][j], dict):
            last_T = max(T[k][j].keys())
            id_T = copy.deepcopy(T[k][j][last_T])
            # res[(max(Ie(B[j], B[1]), list(T[k][j].keys()))[0])] = list(T[k][j].values())[0]
        else:
            continue

        if e(B[j], B[1]) > last_T:
            res[e(B[1], B[j])] = id_T
        else:
            res[last_T] = id_T

    minest = float('inf')
    idx = 0
    # print("res: ", res)
    for key, value in res.items():

        if key < minest:
            minest = key
            idx = value
    return idx


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = np.loadtxt('./dataset/negative_skl.csv', delimiter=',')
    plt.scatter(data[:, 0], data[:, 1], c='gray')

    n = len(data)
    # val, idx = URzero(data, n//3)
    # val, idx = UR(data, n//3)
    # idx = CR(data, n//3)
    idx = ER(data, n//3)
    
    skl = data[idx, :]
    plt.scatter(skl[:, 0], skl[:, 1], c='red')
    plt.show()

    # print(uniformity(data))
    pass
