import numpy as np
import matplotlib.pyplot as plt


def matrix_add(A, B):
    """矩阵加法"""
    return np.add(A, B)


def matrix_mul(A, B):
    """矩阵乘法"""
    return np.dot(A, B)


def matrix_transpose(A):
    """矩阵转置"""
    return np.transpose(A)


def plot_heatmap(mat, title):
    """绘制矩阵热力图"""
    plt.figure()
    plt.imshow(mat, cmap="Blues")
    plt.colorbar()
    plt.title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i,j]:.1f}", ha="center", va="center", color="black")
    plt.show()


if __name__ == "__main__":
    # 示例
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[2, 0, 1], [0, 2, 3]])

    print("===== 基础矩阵运算 =====")
    print("矩阵A：")
    print(A)
    print("矩阵B：")
    print(B)

    add_res = matrix_add(A, B)
    print("\nA+B 的运算结果：")
    print(add_res)
    plot_heatmap(add_res, "A+B 矩阵热力图")

    mul_res = matrix_mul(A, B.T)
    print("\nA × B转置 的运算结果：")
    print(mul_res)
    plot_heatmap(mul_res, "A × B转置 热力图")

    trans_res = matrix_transpose(A)
    print("\n矩阵A的转置：")
    print(trans_res)
    plot_heatmap(trans_res, "矩阵A转置热力图")

    # 方阵特征值求解
    print("\n===== 方阵特征值与特征向量演示 =====")

    square_mat = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    eig_vals, eig_vecs = np.linalg.eig(square_mat)
    print("输入方阵：")
    print(square_mat)
    print("特征值：")
    print(eig_vals)
    print("对应的特征向量（按列）：")
    print(eig_vecs)
    plot_heatmap(square_mat, "用于特征值计算的方阵")

    # SVD奇异值分解演示
    print("\n===== SVD奇异值分解演示 =====")
    U, sigma, VT = np.linalg.svd(A)
    print("原始矩阵A：")
    print(A)
    print("左奇异向量矩阵U：")
    print(U)
    print("奇异值sigma：")
    print(sigma)
    print("右奇异向量转置VT：")
    print(VT)
    # 把一维奇异值扩展成对角矩阵
    sigma_mat = np.zeros_like(A, dtype=float)
    np.fill_diagonal(sigma_mat, sigma)
    plot_heatmap(sigma_mat, "SVD奇异值对角矩阵")
