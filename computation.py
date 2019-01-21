import numpy as np
from dominate import dominate


def BNL(data):
    sky = np.array([])
    del_list = []
    for l, d in enumerate(data):
        # print(l)
        if sky.size == 0:
            sky = np.array([d])
            continue

        for i, s in enumerate(sky):
            if dominate(s, d):  # if being dominated, eliminated.
                break
        else:  # neither can dominate d
            for i, s in enumerate(sky):
                if dominate(d, s):  # if dominate a skyline point, record it.
                    del_list.append(i)
            if del_list:  # delete the dominated skyline points at a time.
                sky = np.delete(sky, del_list, axis=0)
                del_list = []
            # append d to sky.
            sky = np.concatenate((sky, [d]), axis=0)
    return sky


def SFS(data):
    sky = np.array([])
    # TODO SFS
    return sky


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    data = np.loadtxt('./incorrelation.csv', delimiter=',')
    # print(data.shape)
    res = BNL(data[:30000, :])
    # print(res)

    plt.scatter(data[:1000, 0], data[:1000, 1])
    plt.scatter(res[:, 0], res[:, 1], c='red')
    plt.show()
