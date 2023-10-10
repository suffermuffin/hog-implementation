from typing import List, Union, Any
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def img_prep(dir):
    """preparing img"""

    # img into var
    img = cv2.imread(dir)
    # Into gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize
    img = cv2.resize(img, (64, 128))

    return img


def get_nabla_x(p_img):
    """Calculating nabla along x axis"""

    nabla_list_x = np.zeros((p_img.shape[0] - 2, p_img.shape[1] - 2))

    for y in range(1, p_img.shape[0] - 1):
        for x in range(1, p_img.shape[1] - 1):
            num1 = p_img[y][x - 1].astype(float)
            num2 = p_img[y][x + 1].astype(float)

            nabla_x = num1 - num2
            nabla_list_x[y - 1, x - 1] = nabla_x

    # print(nabla_list_x.shape)
    return nabla_list_x


def get_nabla_y(p_img):
    """Calculating nabla along y axis"""

    nabla_list_y = np.zeros((p_img.shape[0] - 2, p_img.shape[1] - 2))

    for x in range(1, p_img.shape[1] - 1):
        for y in range(1, p_img.shape[0] - 1):
            num1 = p_img[y - 1][x].astype(float)
            num2 = p_img[y + 1][x].astype(float)

            nabla_y = num1 - num2
            nabla_list_y[y - 1, x - 1] = nabla_y

            # print(nabla_list_y)
    return nabla_list_y


def get_mu(nab_x, nab_y):
    """Calculates magnitude for the img"""

    mu_list = np.empty((nab_y.shape[0], nab_y.shape[1]))

    for y in range(nab_y.shape[0]):
        for x in range(nab_x.shape[1]):
            mu = math.sqrt((nab_x[y, x] ** 2 + nab_y[y, x] ** 2))
            mu = round(mu, 2)
            mu_list[y, x] = mu

    # print(mu_list)
    return mu_list


def get_fi(nab_x, nab_y):
    """Calculates angel for the img"""

    fi_list = np.empty((nab_y.shape[0], nab_y.shape[1]))

    for y in range(nab_y.shape[0]):
        for x in range(nab_x.shape[1]):
            fi = (180 / math.pi) * (math.atan((nab_y[y, x] / (nab_x[y, x] + 1e-8))) % math.pi)
            fi = round(fi, 2)
            fi_list[y, x] = fi
            # print(fi)

    # print(fi_list)
    return fi_list


def get_figure(img, cell_size=8):
    """Returns cell in 4-dim matrix shape [cell_y, cell_x, pixl_y, pixl_x]"""

    result = np.empty((img.shape[0] // cell_size, img.shape[1] // cell_size, cell_size, cell_size))

    cell_id_x = 0
    cell_id_y = 0

    # fig, axs = plt.subplots(16, 8, sharex=True, sharey=True, figsize=(10, 20))
    for i, y in enumerate(range(0, img.shape[0], cell_size)):
        for j, x in enumerate(range(0, img.shape[1], cell_size)):
            cell = img[y:y + cell_size, x:x + cell_size]
            result[cell_id_y, cell_id_x] = cell

            # axs[i, j].imshow(cell)

            cell_id_x += 1

        cell_id_x = 0
        cell_id_y += 1
    # plt.show()
    return result


def bin_contrib(mu_cell: np.ndarray, fi_cell: np.ndarray) -> np.ndarray:
    """Takes 1 cell of mu and fi. Outputs array of 9bins.
    
    Long description...

    Parameters
    ----------
    mu_cell: ndarray
        numpy array which represents magnitude for one HOG cell.
    fi_cell: ndarray
        numpy array which represents angle for one HOG cell.

    Returns
    -------
    ndarray
        9 bins histogram of oriented gradients for one input cell.
    
    Example
    -------
        # >>> cell = np.array[]
        # >>> bin_conrib(cell)
        4
    """

    w = 20
    mu_bin = np.zeros(9)

    # [0] bin0, [1] bin20, [2] bin40, [3] bin60, [4] bin80, [5] bin100, [6] bin120, [7] bin140, [8] bin160 = 0

    for y in range(mu_cell.shape[0]):
        for x in range(mu_cell.shape[1]):
            fi = fi_cell[y, x]
            mu = mu_cell[y, x]

            fi_r = (fi % w) / w  # Как ФИ относится к правой части в %
            fi_l = 1 - fi_r  # Как ФИ относится к левой части в %

            b_l = (fi // w) * w  # Значение левой корзины

            mu_r = round((mu * fi_r), 2)  # Значение Мью для правой корзины
            mu_l = round((mu * fi_l), 2)  # Значение Мью для левой корзины

            bin_num1 = int(b_l // w)  # Number of a left bin
            bin_num2 = int(bin_num1 + 1)

            if bin_num2 > 8:
                bin_num2 = int(0)

            mu_bin[bin_num1] += mu_l
            mu_bin[bin_num2] += mu_r

    # print(mu_bin)
    return mu_bin


def bin_sint(mu_mat, fi_mat):
    """Takes mu_mat and fi_mat on input and outputs 9_bins_mat"""

    mu_bin = np.zeros((16, 8, 9))

    for y in range(mu_mat.shape[0]):
        for x in range(mu_mat.shape[1]):
            fi_cell = fi_mat[y, x]
            mu_cell = mu_mat[y, x]

            mu_bin[y, x] = bin_contrib(mu_cell, fi_cell)

    return mu_bin


def normal(mat):
    """Normalizes inputed matrix: mat/np.sqrt(np.sum(mat ** 2))"""

    k = np.sqrt(np.sum(mat ** 2))
    mat = mat/(k + 1e-5)
    mat = np.round(mat, 4)

    return mat


def get_block(bin_mat, block_size=4):
    """Concatenates bins coresponding to blocks"""

    full_bin = np.zeros(bin_mat.shape[2]*block_size)

    blocks_mat = np.zeros((bin_mat.shape[0] - 1, bin_mat.shape[1] - 1, bin_mat.shape[2]*block_size))

    for y in range(bin_mat.shape[0]-1):
        for x in range(bin_mat.shape[1]-1):
            full_bin[0:9] = bin_mat[y, x]
            full_bin[9:18] = bin_mat[y, x+1]
            full_bin[18:27] = bin_mat[y+1, x]
            full_bin[27:36] = bin_mat[y+1, x+1]

            blocks_mat[y, x] = normal(full_bin)

            full_bin = np.zeros(bin_mat.shape[2]*block_size)

    return blocks_mat


def get_hog(img):
    """Getting HOG feature 1x3780"""

    padded_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, borderType=0)  # adding padding to img

    nabla_x_img = get_nabla_x(padded_img.astype(float))  #
    nabla_y_img = get_nabla_y(padded_img.astype(float))  # getting gradients alongside axis

    mu = get_mu(nabla_x_img, nabla_y_img)  #
    fi = get_fi(nabla_x_img, nabla_y_img)  # calculating mu & fi

    mu_mat = get_figure(mu)     # 
    fi_mat = get_figure(fi)     # iterarting img into cell grids for mu & fi
    
    bin_mat = bin_sint(mu_mat, fi_mat)  # creating bin for each cell

    block_mat = get_block(bin_mat)      # creating concatenated bin of 36 elements for each block   

    return block_mat.ravel()


def main():
    img = img_prep('Pics/test.jpg')
    hog = get_hog(img)

    print(hog)
    print(len(hog))


if __name__ == '__main__':
    main()