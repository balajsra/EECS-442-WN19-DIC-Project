#############################################################
#             EECS 442: Computer Vision - W19               #
#############################################################
# Authors: Sravan Balaji & Kevin Monpara                    #
# Filename: image_data.py                                   #
# Description:                                              #
#                                                           #
#############################################################

import numpy as np


class ImageData:
    # Displacement Data
    dx = None
    dy = None
    disp_mag = None

    # Strain Data
    eps_x = None
    eps_y = None
    eps_mag = None

    def __init__(self, num_rows, num_cols):
        matrix_shape = (num_rows, num_cols)

        self.dx = np.zeros(matrix_shape)
        self.dy = np.zeros(matrix_shape)
        self.disp_mag = np.zeros(matrix_shape)

        self.eps_x = np.zeros(matrix_shape)
        self.eps_y = np.zeros(matrix_shape)
        self.eps_mag = np.zeros(matrix_shape)
