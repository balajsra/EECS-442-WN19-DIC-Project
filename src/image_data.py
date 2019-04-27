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
    reference_index = None
    comparison_index = None
    location = None
    displacement = None
    strain = None

    def __init__(self, num_rows, num_cols, ref_index, comp_index):
        self.reference_index = ref_index
        self.comparison_index = comp_index

        matrix_shape = (num_rows, num_cols, 2)

        self.location = np.zeros(matrix_shape)
        self.displacement = np.zeros(matrix_shape)
        self.strain = np.zeros(matrix_shape)
