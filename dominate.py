

def dominate(p, q):
    """

    :param p: iterable.
    :param q: iterable.
    :return: True/False
    """
    ltflag = False
    assert len(p) == len(q), 'Dimension must be equal.'
    for i in range(len(p)):
        if p[i] <= q[i]:  # foreach less equal than.
            if p[i] < q[i]:  # at least one less then.
                ltflag = True
        else:
            return False
    if ltflag:
        return True
    else:
        return False


if __name__ == '__main__':

    print(dominate([1, 2], [2, 3]))
    print(dominate([1, 3], [2, 2]))
    print(dominate([1, 2], [1, 2]))
    print(dominate([1.1, 2.3], [1.5, 2.7]))
    print(dominate([1.1, 2.1], [1.1, 2.1]))

    import numpy as np
    print('==================')
    print(dominate(np.array([1, 2]), np.array([2, 3])))
    print(dominate(np.array([1, 3]), np.array([2, 2])))
    print(dominate(np.array([1, 2]), np.array([1, 2])))
    print(dominate(np.array([1.1, 2.3]), np.array([1.5, 2.7])))
    print(dominate(np.array([1.1, 2.1]), np.array([1.1, 2.1])))
