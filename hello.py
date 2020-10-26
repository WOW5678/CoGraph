
# -*- coding:utf-8 -*-
"""
@Time: 2020/10/07 14:41
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""

import numpy as np

arr1 = [[1, 2, 3], [4, 5, 6]]
arr1 = np.array(arr1)

arr2 = [[4, 5, 6], [7,8, 9]]
arr2 = np.array(arr2)

print(np.array([arr1, arr2]).shape)

