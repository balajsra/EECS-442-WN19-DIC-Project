import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import file_data


def main():
	# Read in all images
	images = read_images()

	# Read in data from Section001_Data.txt
	specimen, load_disp_data = file_data.read_file("../Section001_Data.txt")
	
	# Keep track of Stress and Strains
	stresses = []
	strains = []

	# Get distances using sift
	distances = get_sift_distance(images[8], images[9])

	# These distances are coming out as zero for some reason.
	# Still trying to figure out if it's a bug in the code I
	# wrote, or if SIFT won't work for our case.
	print("DISTANCES:")
	print(distances)
	print("MAX DISTANCE:")
	print(max(distances))
	print("MIN DISTANCE:")
	print(min(distances))

	# Eventually we'll find the distances, stress, strain for all images
	"""
	for idx in range(0, len(images)-1):
		distances = getSiftDistance(images[idx], images[idx+1])
		strains.append(getStrain(specimen.ol, load_disp_data[idx].disp))
		stresses.append(load_disp_data[idx].stress)
		youngs_mod = getYoungsModulus(strains[idx] / stress[idx])
	"""


def read_images():
	image_dir = '../Images/'
	filenames = os.listdir(image_dir)

	images = []

	images.append(None)

	for file in filenames:
		images.append(cv2.imread(os.path.join(image_dir, file), 0))

	return images


def get_strain(length, displacement):
	return displacement / length


def get_youngs_modulus(strain, stress):
	return stress / strain


def get_sift_distance(img1, img2):
	""" Gets distance between matching pts in
		img1 and img2.
	"""
	sift = cv2.xfeatures2d.SIFT_create()
	original_kp, original_des = sift.detectAndCompute(img1, None)
	new_kp, new_des = sift.detectAndCompute(img2, None)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(original_des, new_des, k=2)

	# Ratio test
	good = []

	for m, n in matches:
		if m.distance < 0.3 * n.distance:
			good.append(m)

	# Draw matches
	"""
	# Uncomment to print matches between img1 and img2.
	# SIFT may not be the best method based off the matched
	# images.
	matches = cv2.drawMatchesKnn(img1, original_kp, img2, new_kp, good, None, flags=2)
	plt.imshow(matches)
	plt.show()
	"""

	# Featured matched keypoints from images 1 and 2
	pts1 = np.float32([original_kp[m.queryIdx].pt for m in good])
	pts2 = np.float32([new_kp[m.trainIdx].pt for m in good])

	# convert to complex number
	z1 = np.array([[complex(c[0],c[1]) for c in pts1]])
	z2 = np.array([[complex(c[0],c[1]) for c in pts2]])

	# Distance between featured matched keypoints
	distances = abs(z2 - z1)

	return distances


def find_displacement(match_method):
	images = read_images()

	specimen, load_disp_data = file_data.read_file("../Section001_Data.txt")

	plt.figure(1)

	reference = images[8]
	compare_img = images[12]

	plt.imshow(reference, cmap="gray", vmin=0, vmax=255)

	subset_size = 5
	subset_spacing = 20

	for x in range(650, 2080, subset_spacing):
		for y in range(120, 500, subset_spacing):
			up_bound = (subset_size + 1) // 2
			low_bound = subset_size // 2

			subset = reference[y-low_bound:y+up_bound, x-low_bound:x+up_bound]

			res = cv2.matchTemplate(image=compare_img, templ=subset, method=match_method)

			minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)

			dx = None
			dy = None

			if match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
				dx = minLoc[0] - x
				dy = minLoc[1] - y
			elif match_method in [cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
			                      cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
				dx = maxLoc[0] - x
				dy = maxLoc[1] - y

			plt.arrow(x=x, y=y, dx=dx, dy=dy, color="yellow", length_includes_head=True, shape="full")

	plt.show()


if __name__ == '__main__':
	# main()
	for match_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED,
	                     cv2.TM_CCORR, cv2.TM_CCORR_NORMED,
	                     cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED]:
		find_displacement(match_method)
