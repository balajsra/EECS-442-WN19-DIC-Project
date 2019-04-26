import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import file_data
import image_data


def read_images():
    image_dir = '../Images/'
    filenames = os.listdir(image_dir)

    images = []

    images.append(None)

    for file in filenames:
        images.append(cv2.imread(os.path.join(image_dir, file), 0))

    return images


def find_displacement(match_method):
    images = read_images()

    specimen, load_disp_data = file_data.read_file("../Section001_Data.txt")

    plt.figure(1)

    reference = images[7]
    compare_img = images[8]

    plt.imshow(reference, cmap="gray", vmin=0, vmax=255)

    subset_size = 5
    subset_spacing = 30
    search_size = 5

    x_range = range(650, 2080, subset_spacing)
    y_range = range(120, 500, subset_spacing)

    im_data = image_data.ImageData(len(y_range), len(x_range))

    disp_mag = np.zeros((len(y_range), len(x_range)))

    for i in range(0, len(y_range)):
        for j in range(0, len(x_range)):
            x = x_range[j]
            y = y_range[i]

            im_data.location[i, j, :] = np.array([x, y])

            up_bound = (subset_size + 1) // 2
            low_bound = subset_size // 2

            subset = reference[y-low_bound:y+up_bound, x-low_bound:x+up_bound]
            search = compare_img[y-search_size:y+search_size+1, x-search_size:x+search_size+1]

            res = cv2.matchTemplate(image=search, templ=subset, method=match_method)

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(src=res)

            dx = None
            dy = None

            if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                dx = (minLoc[0] + (subset_size // 2)) - search_size
                dy = (minLoc[1] + (subset_size // 2)) - search_size
            elif match_method in [cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
                                  cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
                dx = (maxLoc[0] + (subset_size // 2)) - search_size
                dy = (maxLoc[1] + (subset_size // 2)) - search_size

            im_data.displacement[i, j, :] = np.array([dx, dy])
            disp_mag[i, j] = np.sqrt((dx ** 2) + (dy ** 2))

    # Strain in x-direction
    im_data.strain[:, :, 0] = np.gradient(
        im_data.displacement[:, :, 0],      # x displacements
        im_data.location[:, 0, 1],          # y (row) positions
        im_data.location[0, :, 0])[1]       # x (col) positions

    im_data.strain[:, :, 1] = np.gradient(
        im_data.displacement[:, :, 1],      # y displacements
        im_data.location[:, 0, 1],          # y (row) positions
        im_data.location[0, :, 0])[0]       # x (col) positions

    # plt.quiver(X=im_data.location[0, :, 0], Y=im_data.location[:, 0, 1],
    #            U=im_data.strain[:, :, 0], V=im_data.strain[:, :, 1],
    #            C=strain_mag, cmap=plt.cm.jet)

    plt.quiver(im_data.location[:, :, 0],       # x coordinates of arrow locations
               im_data.location[:, :, 1],       # y coordinates of arrow locations
               im_data.displacement[:, :, 0],   # x components of arrow vectors
               im_data.displacement[:, :, 1],   # y components of arrow vectors
               disp_mag,                        # arrow color (vector magnitude)
               cmap=plt.cm.jet)                 # color map (jet)

    plt.show()


if __name__ == '__main__':
    find_displacement(cv2.TM_SQDIFF_NORMED)

    # for match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED,
    #                      cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
    #                      cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
    #     find_displacement(match_method)
