#############################################################
#             EECS 442: Computer Vision - W19               #
#############################################################
# Authors: Sravan Balaji & Kevin Monpara                    #
# Filename: __main__.py                                     #
# Description:                                              #
#   Perform DIC and generate various plots for Young's      #
#   modulus, Poisson's ratio, matching method comparisons,  #
#   etc.                                                    #
#############################################################

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import file_data
import image_data

PIXELS_TO_MM = None


def read_images():
    # Read images into numpy arrays
    image_dir = '../Images/'
    filenames = os.listdir(image_dir)

    images = []

    images.append(None)

    for file in filenames:
        images.append(cv2.imread(os.path.join(image_dir, file), 0))

    return images


def find_displacement(images, ref_index, comp_index, match_method,
                      ref_template_size, search_window_size, grid_spacing):
    # Compare subsets from reference image to compare_img to find displacement and strain
    reference = images[ref_index]
    compare_img = images[comp_index]

    x_range = range(200, 2200, grid_spacing)
    y_range = range(150, 450, grid_spacing)

    im_data = image_data.ImageData(len(y_range), len(x_range), ref_index, comp_index,
                                   ref_template_size, search_window_size, grid_spacing)

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

    disp_matrix = np.zeros((ref_img.shape[0], ref_img.shape[1], 2))
    strain_matrix = np.zeros((ref_img.shape[0], ref_img.shape[1], 2))
    rows, cols, channels = img_data.location.shape

    # Expand displacement and strain values into larger pixel windows for visualization
    for r in range(rows):
        for c in range(cols):
            row_start = int(img_data.location[0, 0, 1] + r * img_data.grid_spacing)
            col_start = int(img_data.location[0, 0, 0] + c * img_data.grid_spacing)

            row_range = [row_start, row_start + img_data.grid_spacing]
            col_range = [col_start, col_start + img_data.grid_spacing]

            disp_matrix[row_range[0]:row_range[1], col_range[0]:col_range[1], 0].fill(img_data.displacement[r, c, 0])
            disp_matrix[row_range[0]:row_range[1], col_range[0]:col_range[1], 1].fill(img_data.displacement[r, c, 1])
            strain_matrix[row_range[0]:row_range[1], col_range[0]:col_range[1], 0].fill(img_data.strain[r, c, 0])
            strain_matrix[row_range[0]:row_range[1], col_range[0]:col_range[1], 1].fill(img_data.strain[r, c, 1])

    # Subplot 1: Axial Displacement
    ax = plt.subplot(2, 2, 1)

    ax.title.set_text("Axial Displacement between image " + str(img_data.reference_index)
                      + " and image " + str(img_data.comparison_index))

    plt.imshow(
        ref_img,        # Show reference image
        cmap="gray",    # Grayscale
        vmin=0,         # Minimum pixel value
        vmax=255,       # Maximum pixel value
        origin="lower"  # Flip image so increasing row corresponds to increasing y
    )

    mask = np.ma.masked_where(disp_matrix[:, :, 0] == 0, disp_matrix[:, :, 0] * -PIXELS_TO_MM)
    plt.imshow(mask, cmap="jet", interpolation="none", origin="lower")

    plt.colorbar(ax=ax)

    # Subplot 2: Transverse Displacement
    ax = plt.subplot(2, 2, 2)

    ax.title.set_text("Transverse Displacement between image " + str(img_data.reference_index)
                      + " and image " + str(img_data.comparison_index))

    ax.imshow(
        ref_img,        # Show reference image
        cmap="gray",    # Grayscale
        vmin=0,         # Minimum pixel value
        vmax=255,       # Maximum pixel value
        origin="lower"  # Flip image so increasing row corresponds to increasing y
    )

    mask = np.ma.masked_where(disp_matrix[:, :, 1] == 0, disp_matrix[:, :, 1] * PIXELS_TO_MM)
    ax.imshow(mask, cmap="jet", interpolation="none", origin="lower")

    plt.colorbar(ax=ax)

    # Subplot 3: Axial Strain
    ax = plt.subplot(2, 2, 3)

    ax.title.set_text("Axial Strain between image " + str(img_data.reference_index)
                      + " and image " + str(img_data.comparison_index))

    ax.imshow(
        ref_img,        # Show reference image
        cmap="gray",    # Grayscale
        vmin=0,         # Minimum pixel value
        vmax=255,       # Maximum pixel value
        origin="lower"  # Flip image so increasing row corresponds to increasing y
    )

    mask = np.ma.masked_where(strain_matrix[:, :, 0] == 0, strain_matrix[:, :, 0] * -1)
    ax.imshow(mask, cmap="jet", interpolation="none", origin="lower")

    plt.colorbar(ax=ax)

    # Subplot 4: Transverse Strain
    ax = plt.subplot(2, 2, 4)

    ax.title.set_text("Transverse Strain between image " + str(img_data.reference_index)
                      + " and image " + str(img_data.comparison_index))

    ax.imshow(
        ref_img,        # Show reference image
        cmap="gray",    # Grayscale
        vmin=0,         # Minimum pixel value
        vmax=255,       # Maximum pixel value
        origin="lower"  # Flip image so increasing row corresponds to increasing y
    )

    mask = np.ma.masked_where(strain_matrix[:, :, 1] == 0, strain_matrix[:, :, 1])
    ax.imshow(mask, cmap="jet", interpolation="none", origin="lower")

    plt.colorbar(ax=ax)

    plt.show()


def compare_matching_quality(images, ref_index, comp_index,
                             ref_template_size, search_window_size, grid_spacing):
    # Compare Quality of Displacement Tracking Methods
    plt.figure()
    plt.suptitle("Displacements between image " + str(ref_index) + " and image " + str(comp_index))

    i = 1
    for match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED,
                         cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
                         cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
        reference_img, img_data = find_displacement(
            images=images,
            ref_index=ref_index,
            comp_index=comp_index,
            match_method=match_method,
            ref_template_size=ref_template_size,
            search_window_size=search_window_size,
            grid_spacing=grid_spacing
        )

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

        plt.colorbar(ax=ax)

        i += 1

    plt.show()


def compare_matching_time():
    # Compare Runtime of Displacement Tracking Methods
    plt.figure()
    plt.title("Matching Method Runtime Comparison")

    methods = ["SQDIFF", "SQDIFF_NORMED", "CCORR", "CCORR_NORMED", "CCOEFF", "CCOEFF_NORMED"]

    # Runtime in seconds from 3 trials
    SQDIFF = [6.969048976898193, 6.824753761291504, 6.946980714797974]
    SQDIFF_NORMED = [7.051121950149536, 7.040151357650757, 6.851653814315796]
    CCORR = [7.0052430629730225, 6.867607116699219, 6.804780006408691]
    CCORR_NORMED = [7.036186933517456, 6.773889064788818, 6.812784910202026]
    CCOEFF = [6.799818754196167, 7.141877889633179, 6.904510736465454]
    CCOEFF_NORMED = [6.861653566360474, 6.8746209144592285, 7.113796710968018]

    x_pos = np.arange(len(methods))
    means = [np.mean(SQDIFF), np.mean(SQDIFF_NORMED),
             np.mean(CCORR), np.mean(CCORR_NORMED),
             np.mean(CCOEFF), np.mean(CCOEFF_NORMED)]
    error = [np.std(SQDIFF), np.std(SQDIFF_NORMED),
             np.std(CCORR), np.std(CCORR_NORMED),
             np.std(CCOEFF), np.std(CCOEFF_NORMED)]

    plt.bar(x_pos, means, yerr=error, align="center", alpha=0.5, ecolor="black", capsize=10)
    plt.xticks(x_pos, methods)

    plt.show()


def plot_frame_data(frame_data):
    # Use data from LabVIEW file to plot displacement, load, and stress curves
    plt.figure()

    x = []
    y_1 = []
    y_2 = []
    y_3 = []

    for i in range(1, len(frame_data)):
        x.append(i)
        y_1.append(frame_data[i].disp)
        y_2.append(frame_data[i].load / 1000)   # Convert from N to kN
        y_3.append(frame_data[i].stress)

    ax = plt.subplot(3, 1, 1)
    ax.plot(x, y_1)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Axial Displacement (mm)")

    ax = plt.subplot(3, 1, 2)
    ax.plot(x, y_2)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Axial Load (kN)")

    ax = plt.subplot(3, 1, 3)
    ax.plot(x, y_3)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Axial Stress (MPa)")

    plt.show()


def stress_strain_curve(specimen, frame_data, strain):
    # Plot stress strain curve using LabVIEW file and DIC to compare results
    plt.figure()

    x = []
    y = []

    for i in range(1, len(frame_data)):
        x.append(frame_data[i].disp / specimen.ol)
        y.append(frame_data[i].stress)

    ax = plt.subplot(2, 1, 1)
    ax.title.set_text("Stress-Strain Curve (Overall Length)")
    plt.plot(x, y)
    plt.xlabel("Axial Strain")
    plt.ylabel("Axial Stress (MPa)")

    ax = plt.subplot(2, 1, 2)
    ax.title.set_text("Stress-Strain Curve (Gauge Length)")

    plt.plot(strain[2:], y[2:659])
    plt.xlabel("Axial Strain")
    plt.ylabel("Axial Stress (MPa)")

    plt.show()


def strain_measurement(images, match_method, ref_template_size, search_window_size, pt1, pt2):
    # Pick 2 points on image, track their displacements, compute the strain of the line connecting them
    orig_len_x = np.abs(pt2[0] - pt1[0])
    orig_len_y = np.abs(pt2[1] - pt1[1])

    length_x = np.zeros((659,))
    length_y = np.zeros((659,))

    for i in range(1, 658):
        reference = images[i]
        compare_img = images[i + 1]

        for pt in [pt1, pt2]:
            x, y = pt

            template_x = [x - (ref_template_size[0] // 2), x + ((ref_template_size[0] + 1) // 2)]
            template_y = [y - (ref_template_size[1] // 2), y + ((ref_template_size[1] + 1) // 2)]
            search_x = [template_x[0] - search_window_size[0], template_x[1] + search_window_size[0]]
            search_y = [template_y[0] - search_window_size[1], template_y[1] + search_window_size[1]]

            ref_template = reference[template_y[0]:template_y[1], template_x[0]:template_x[1]]
            search_window = compare_img[search_y[0]:search_y[1], search_x[0]:search_x[1]]

            res = cv2.matchTemplate(
                image=search_window,  # Search window in compare image
                templ=ref_template,  # Template from reference image
                method=match_method  # Matching Method
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

            if pt == pt1:
                pt1[0] += dx
                pt1[1] += dy
            elif pt == pt2:
                pt2[0] += dx
                pt2[1] += dy

        length_x[i + 1] = np.abs(pt2[0] - pt1[0])
        length_y[i + 1] = np.abs(pt2[1] - pt1[1])

    strain_x = (length_x - orig_len_x) / orig_len_x
    strain_y = (length_y - orig_len_y) / orig_len_y

    return strain_x, strain_y


def poisson_ratio(strain_x, strain_y):
    # Plot Transverse vs. Axial strain
    plt.figure()
    plt.title("Transverse vs. Axial Strain")
    plt.xlabel("Axial Strain")
    plt.ylabel("Transverse Strain")
    plt.plot(strain_x[2:], strain_y[2:])

    plt.show()


if __name__ == '__main__':
    # Read in images from Images folder
    images = read_images()

    # Read in numerical data from LabVIEW output file
    specimen, frame_data = file_data.read_file("../Section001_Data.txt")

    # Width = 16.61 mm = 404 pixels
    PIXELS_TO_MM = specimen.w / 404

    # Plot data from LabVIEW file
    plot_frame_data(frame_data)

    # Get axial and transverse strain measurements using 2 points
    strain_x, strain_y = strain_measurement(
        images=images,
        match_method=cv2.TM_CCORR_NORMED,
        ref_template_size=(9, 9),
        search_window_size=(5, 3),
        pt1=[725, 200],
        pt2=[1975, 400]
    )

    # Plot Stress-Strain curve from LabVIEW data as well as
    stress_strain_curve(specimen, frame_data, strain_x)

    # Plot Transverse vs Axial strain to get Poisson's Ratio
    poisson_ratio(strain_x, strain_y)

    # Plot runtime of different matching methods
    compare_matching_time()

    # Plot displacement using different matching methods
    compare_matching_quality(
        images=images,
        ref_index=8,
        comp_index=9,
        ref_template_size=(5, 5),
        search_window_size=(10, 5),
        grid_spacing=5
    )
    compare_matching_quality(
        images=images,
        ref_index=560,
        comp_index=561,
        ref_template_size=(5, 5),
        search_window_size=(10, 5),
        grid_spacing=5
    )

    # Matching Methods available in OpenCV
    #   cv2.TM_SQDIFF
    #   cv2.TM_SQDIFF_NORMED
    #   cv2.TM_CCORR
    #   cv2.TM_CCORR_NORMED
    #   cv2.TM_CCOEFF
    #   cv2.TM_CCOEFF_NORMED

    # Generate displacement and strain plots at regular intervals in tensile test
    for i in range(8, 680, 25):
        reference_img, img_data = find_displacement(
            images=images,
            ref_index=i,
            comp_index=i+5,
            match_method=cv2.TM_CCORR_NORMED,
            ref_template_size=(5, 5),
            search_window_size=(10, 5),
            grid_spacing=5
        )

        plot_disp_and_strain(reference_img, img_data)

    # Generate displacement and strain plot for image right before fracture
    reference_img, img_data = find_displacement(
        images=images,
        ref_index=657,
        comp_index=658,
        match_method=cv2.TM_CCORR_NORMED,
        ref_template_size=(5, 5),
        search_window_size=(10, 5),
        grid_spacing=5
    )

    plot_disp_and_strain(reference_img, img_data)
