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
    ref_template_size = None
    search_window_size = None
    grid_spacing = None
    location = None
    displacement = None
    strain = None

    def __init__(self, num_rows, num_cols,
                 ref_index, comp_index,
                 ref_template_size,
                 search_window_size,
                 grid_spacing):
        self.reference_index = ref_index
        self.comparison_index = comp_index

        self.ref_template_size = ref_template_size
        self.search_window_size = search_window_size
        self.grid_spacing = grid_spacing

        matrix_shape = (num_rows, num_cols, 2)

        self.location = np.zeros(matrix_shape)
        self.displacement = np.zeros(matrix_shape)
        self.strain = np.zeros(matrix_shape)
