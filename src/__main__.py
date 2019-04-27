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


def find_displacement(images, ref_index, comp_index, match_method):
    reference = images[ref_index]
    compare_img = images[comp_index]

    ref_template_size = (5, 5)
    search_window_size = (5, 3)
    grid_spacing = 10

    x_range = range(200, 2200, grid_spacing)
    y_range = range(100, 500, grid_spacing)

    im_data = image_data.ImageData(len(y_range), len(x_range), ref_index, comp_index)

    for i in range(len(y_range)):
        for j in range(len(x_range)):
            x = x_range[j]
            y = y_range[i]

            im_data.location[i, j, :] = np.array([x, y])

            template_x = [x - (ref_template_size[0] // 2), x + ((ref_template_size[0] + 1) // 2)]
            template_y = [y - (ref_template_size[1] // 2), y + ((ref_template_size[1] + 1) // 2)]
            search_x = [template_x[0] - search_window_size[0], template_x[1] + search_window_size[0]]
            search_y = [template_y[0] - search_window_size[1], template_y[1] + search_window_size[1]]

            ref_template = reference[template_y[0]:template_y[1], template_x[0]:template_x[1]]
            search_window = compare_img[search_y[0]:search_y[1], search_x[0]:search_x[1]]

            res = cv2.matchTemplate(
                image=search_window,    # Search window in compare image
                templ=ref_template,     # Template from reference image
                method=match_method     # Matching Method
            )

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src=res)

            dx = None
            dy = None

            if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                dx = min_loc[0] - search_window_size[0]
                dy = min_loc[1] - search_window_size[1]
            elif match_method in [cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
                                  cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
                dx = max_loc[0] - search_window_size[0]
                dy = max_loc[1] - search_window_size[1]

            im_data.displacement[i, j, :] = np.array([dx, dy])

    # Strain in x direction
    im_data.strain[:, :, 0] = np.gradient(
        im_data.displacement[:, :, 0],      # x displacements
        im_data.location[:, 0, 1],          # y (row) positions
        im_data.location[0, :, 0]           # x (col) positions
    )[1]                                    # Gradient in cols direction

    # Strain in y direction
    im_data.strain[:, :, 1] = np.gradient(
        im_data.displacement[:, :, 1],      # y displacements
        im_data.location[:, 0, 1],          # y (row) positions
        im_data.location[0, :, 0]           # x (col) positions
    )[0]                                    # Gradient in rows direction

    return reference, im_data


def plot_disp_and_strain(ref_img, img_data):
    plt.figure()

    # Subplot 1: Displacement
    ax = plt.subplot(2, 1, 1)
    ax.title.set_text("Displacements between image " + str(img_data.reference_index)
                      + " and image " + str(img_data.comparison_index))

    plt.imshow(ref_img,         # Show reference image
               cmap="gray",     # Grayscale
               vmin=0,          # Minimum pixel value
               vmax=255,        # Maximum pixel value
               origin="lower")  # Flip image so increasing row corresponds to increasing y

    disp_mag = np.sqrt(img_data.displacement[:, :, 0] ** 2 + img_data.displacement[:, :, 1] ** 2)

    plt.quiver(img_data.location[:, :, 0],      # x coordinates of arrow locations
               img_data.location[:, :, 1],      # y coordinates of arrow locations
               img_data.displacement[:, :, 0],  # x components of arrow vectors
               img_data.displacement[:, :, 1],  # y components of arrow vectors
               disp_mag,                        # arrow color (vector magnitude)
               cmap=plt.cm.jet,                 # color map (jet)
               units="dots",                    # units of arrow dimensions
               angles="xy")                     # arrows point from (x, y) to (x + u, y + v)

    # Subplot 2: Strain
    ax = plt.subplot(2, 1, 2)
    ax.title.set_text("Strain between image " + str(img_data.reference_index)
                      + " and image " + str(img_data.comparison_index))

    plt.imshow(ref_img,         # Show reference image
               cmap="gray",     # Grayscale
               vmin=0,          # Minimum pixel value
               vmax=255,        # Maximum pixel value
               origin="lower")  # Flip image so increasing row corresponds to increasing y

    strain_mag = np.sqrt(img_data.strain[:, :, 0] ** 2 + img_data.strain[:, :, 1] ** 2)

    plt.quiver(img_data.location[0, :, 0],
               img_data.location[:, 0, 1],
               img_data.strain[:, :, 0],
               img_data.strain[:, :, 1],
               strain_mag,
               cmap=plt.cm.jet,
               units="dots",
               angles="xy")

    plt.show()


def compare_matching_methods(images, ref_idx, comp_idx):
    plt.figure()
    plt.suptitle("Displacements between image " + str(ref_idx) + " and image " + str(comp_idx))

    i = 1
    for match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED,
                         cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
                         cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
        reference_img, img_data = find_displacement(images, ref_idx, comp_idx, match_method)

        ax = plt.subplot(3, 2, i)

        title = None

        if match_method == cv2.TM_SQDIFF:
            title = "SQDIFF"
        elif match_method == cv2.TM_SQDIFF_NORMED:
            title = "SQDIFF_NORMED"
        elif match_method == cv2.TM_CCORR:
            title = "CCORR"
        elif match_method == cv2.TM_CCORR_NORMED:
            title = "CCORR_NORMED"
        elif match_method == cv2.TM_CCOEFF:
            title = "CCOEFF"
        elif match_method == cv2.TM_CCOEFF_NORMED:
            title = "CCOEFF_NORMED"

        ax.title.set_text(title)

        plt.imshow(reference_img, cmap="gray", vmin=0, vmax=255, origin="lower")

        disp_mag = np.sqrt(img_data.displacement[:, :, 0] ** 2 + img_data.displacement[:, :, 1] ** 2)

        plt.quiver(img_data.location[:, :, 0], img_data.location[:, :, 1], img_data.displacement[:, :, 0],
                   img_data.displacement[:, :, 1], disp_mag, cmap=plt.cm.jet, units="dots", angles="xy")

        i += 1

    plt.show()


if __name__ == '__main__':
    images = read_images()

    specimen, load_disp_data = file_data.read_file("../Section001_Data.txt")

    compare_matching_methods(images, 560, 561)

    # Matching Methods
    #   cv2.TM_SQDIFF
    #   cv2.TM_SQDIFF_NORMED
    #   cv2.TM_CCORR
    #   cv2.TM_CCORR_NORMED
    #   cv2.TM_CCOEFF
    #   cv2.TM_CCOEFF_NORMED
    reference_img, img_data = find_displacement(images, 560, 561, cv2.TM_CCORR_NORMED)
    plot_disp_and_strain(reference_img, img_data)
