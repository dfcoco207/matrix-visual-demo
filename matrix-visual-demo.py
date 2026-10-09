import numpy as np


def matrix_add(A, B):
    """矩阵加法"""
    return np.add(A, B)


def matrix_mul(A, B):
    """矩阵乘法"""
    return np.dot(A, B)


def matrix_transpose(A):
    """矩阵转置"""
    return np.transpose(A)


if __name__ == "__main__":
    # 自定义示例矩阵
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[2, 0, 1], [0, 2, 3]])

    print("矩阵A：")
    print(A)
    print("矩阵B：")
    print(B)

    add_res = matrix_add(A, B)
    print("\nA+B 的运算结果：")
    print(add_res)

    mul_res = matrix_mul(A, B.T)
    print("\nA × B转置 的运算结果：")
    print(mul_res)

    trans_res = matrix_transpose(A)
    print("\n矩阵A的转置：")
    print(trans_res)
