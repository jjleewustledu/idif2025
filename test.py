from __future__ import absolute_import
from __future__ import print_function
from multiprocessing import Pool
from six.moves import range


def some_func(a, b):
    return a + b


if __name__ == '__main__':
    data = 10
    inputs = list(range(1, 11))  # this generates a list: [1, 2, 3, ..., 10]

    with Pool() as pool:
        results = pool.starmap(some_func, [(x, data) for x in inputs])

    print(results)
