import numpy as np
import cv2


def sift_distance():
    sift = cv2.xfeatures2d.SIFT_create()
    original_kp, original_des = sift.detectAndCompute(left,None)
    new_kp, new_des = sift.detectAndCompute(right,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(original_des, new_des, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            good.append([m])

    # Featured matched keypoints from images 1 and 2
    pts1 = np.float32([original_kp[m.queryIdx].pt for m in good])
    pts2 = np.float32([new_kp[m.trainIdx].pt for m in good])

    #convert to complex number
    z1 = np.array([[complex(c[0],c[1]) for c in pts1]])
    z2 = np.array([[complex(c[0],c[1]) for c in pts2]])

    # Distance between featured matched keypoints
    FM_dist = abs(z2 - z1)
