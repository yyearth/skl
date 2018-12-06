import numpy as np
import cv2
import random
import requests


# print(cv2.__version__)
# print(np.__version__)

class DataGen(object):

    def __init__(self, dim=2, data=None):
        self.dim = dim
        self.data = data

    def gen(self, dim, n, range_=None):
        """

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

        # for i in range(n):
        #     one_sample = []
        #     for r in range_:
        #         d = random.uniform(*r)
        #         # round(number[, ndigits]) -> number
        #         da = round(d, 1)
        #         one_sample.append(da)
        #     self.data.append(one_sample)

        d = np.random.rand(n, dim)
        m = np.array([])
        a = np.array([])
        for i, r in enumerate(range_):
            l, u = r
            mul = (u - l) * np.ones((n, 1))
            add = u * np.ones((n, 1))
            if i == 0:
                m = mul
                a = add
            else:
                m = np.concatenate((m, mul), axis=1)
                a = np.concatenate((a, add), axis=1)
        self.data = np.round(m * d + a, 1)
        return self.data

    @staticmethod
    def mouse(event, x, y, flags, param):
        # print(x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            print((x, y))
            

    def gen2d(self, n=None, range_=None):

        cv2.namedWindow('DataGen')
        cv2.setMouseCallback('DataGen', DataGen.mouse, self)
        img = np.ones((600, 600)) * 255

        while True:
            cv2.imshow('DataGen', img)
            if cv2.waitKey(1) == 27:
                break


if __name__ == '__main__':
    dg = DataGen()
    # d = dg.gen(3, 4)
    # print(d)
    dg.gen2d()
