import numpy as np
import cv2
import time


# print(cv2.__version__)
# print(np.__version__)
# 3.4.3 cv
# 1.15.1 numpy

def linscale(d, range_):
    """
    Linear scaling.
    :param d: data number in range between 0 to 1.
    :param range_: scale in each dimension.
    :return: scaled data.
    """
    # assert len(d) == len(range_)
    m = np.array([])
    a = np.array([])
    for i, r in enumerate(range_):
        if len(r) != 2:
            continue
        l, u = r
        mul = (u - l) * np.ones((len(d), 1))
        add = l * np.ones((len(d), 1))
        if i == 0:
            m = mul
            a = add
        else:
            m = np.concatenate((m, mul), axis=1)
            a = np.concatenate((a, add), axis=1)

    return np.round(m * d + a, 1)


class DataGen(object):
    __tmpdata = []
    WIDTH = 600
    HEIGHT = 600

    def __init__(self, dim=2, data=None):
        self.dim = dim
        self.data = data

    def gen(self, n, dim, range_=None):
        """
        Generate data.
        :param dim: dimension of data.
        :param n: num of sample data.
        :param range_: range in each dim. e.g. [[a, b], [c, d],...].
                       Default is [0, 10].
        :return: generated data.
        """
        # assert len(range_) == dim, 'Dimension must equal to length of range.'
        assert self.data is None, 'Data is not empty.'

        # If not specified range of each dimension, generate default [0, 10].
        if range_ is None:
            range_ = [[0, 10]] * dim

        self.data = []

        d = np.random.rand(n, dim)
        self.data = linscale(d, range_)

        return self.data

    @staticmethod
    def mouse(event, x, y, flags, param):
        img, tmpdata = param
        if event == cv2.EVENT_LBUTTONDOWN:
            tmpdata.append([x / DataGen.HEIGHT, (DataGen.WIDTH - y) / DataGen.WIDTH])
            cv2.circle(img, (x, y), 2, (0, 0, 200), -1)

    def gen2d(self, n=None, range_=None):
        """
        Generate 2-dim data with cursor.
        :param n: number of point.
        :param range_: Range of X and Y.
        :return: generated data.
        """

        if range_ is None:
            range_ = [[0, 10]] * 2

        assert len(range_) == 2, 'Length of param range_ must be 2.'

        img = np.ones((DataGen.HEIGHT, DataGen.WIDTH, 3)) * 255
        cv2.namedWindow('DataGen')
        cv2.setMouseCallback('DataGen', DataGen.mouse, (img, DataGen.__tmpdata))

        while True:
            cv2.imshow('DataGen', img)
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

        d = np.array(DataGen.__tmpdata)
        self.data = linscale(d, range_)

        return self.data

    def plot(self):
        pass

    def save(self, name=None):
        if name is None:
            name = './data{}.txt'.format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))

        np.savetxt(name, self.data)

    def load(self):
        pass

    def dump(self, n):
        pass


if __name__ == '__main__':
    dg = DataGen()
    d = dg.gen(4, 3, [[1, 2], [10, 20], [12, 13]])
    # d = dg.gen2d()
    print(d)
    dg.save()
