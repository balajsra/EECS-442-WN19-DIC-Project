import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import file_data

def main():

	#Read in all images
	images = readImages()

	#Read in data from Section001_Data.txt
	specimen, load_disp_data = file_data.read_file("../Section001_Data.txt")
	
	#Keep track of Stress and Strains
	stresses = []
	strains = []

	#Get distances using sift
	distances = getSiftDistance(images[0], images[1])

	# These distances are coming out as zero for some reason.
	# Still trying to figure out if it's a bug in the code I
	# wrote, or if SIFT won't work for our case.
	print("DISTANCES:")
	print(distances)
	print("MAX DISTANCE:")
	print(max(distances))
	print("MIN DISTANCE:")
	print(min(distances))

	#Eventually we'll find the distances, stress, strain for all images
	"""
	for idx in range(0, len(images)-1):
		distances = getSiftDistance(images[idx], images[idx+1])
		strains.append(getStrain(specimen.ol, load_disp_data[idx].disp))
		stresses.append(load_disp_data[idx].stress)
		youngs_mod = getYoungsModulus(strains[idx] / stress[idx])
	"""


def readImages():
	image_dir = '../images/'
	filenames = os.listdir(image_dir)

	images = []
	for file in filenames:
		images.append(cv2.imread(os.path.join(image_dir,file)))
	return images

def getStrain(length, displacement):
	return displacement / length

def getYoungsModulus(strain, stress):
	return strain / stress

def getSiftDistance(img1, img2):
	""" Gets distance between matching pts in
		img1 and img2.
	"""
	sift = cv2.xfeatures2d.SIFT_create()
	original_kp, original_des = sift.detectAndCompute(img1,None)
	new_kp, new_des = sift.detectAndCompute(img2,None)

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

	#convert to complex number
	z1 = np.array([[complex(c[0],c[1]) for c in pts1]])
	z2 = np.array([[complex(c[0],c[1]) for c in pts2]])

	# Distance between featured matched keypoints
	distances = abs(z2 - z1)

	return distances




if __name__ == '__main__':
	main()