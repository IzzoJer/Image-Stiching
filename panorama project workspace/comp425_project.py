import numpy as np
import cv2 as cv
import random

img_boxes = 'project_images/Boxes.png'
img_rainier1 = 'project_images/Rainier1.png'
img_rainier2 = 'project_images/Rainier2.png'


# STEP 1 USING BUILD IN SIFT TO DETECT CORNERS AND DISPLAY KEY POINTS

# takes in a src_img(image name) and the name for the output image
def detect_corners(src_img, out_name):
    img = cv.imread(src_img)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    kp = sift.detect(img_gray, None)
    img = cv.drawKeypoints(img_gray, kp, img)
    cv.imwrite(out_name, img)
    cv.imshow(out_name, img)
    cv.waitKey(0)


# CREATES IMAGE FOR boxes.png IN 1a.png
detect_corners(img_boxes, '1a.png')

# CREATES IMAGE FOR Rainier1.png IN 1b.png
detect_corners(img_rainier1, '1b.png')

# CREATES IMAGE FOR Rainier2.png IN 1c.png
detect_corners(img_rainier2, '1c.png')


# END OF STEP 1


# STEP 2 MATCHING INTEREST POINTS USING BUILD IN SIFT TO DISPLAY ALL MATCHES

# takes in 2 images and outputs their key points and matches
def kp_matcher(img1, img2):
    img1 = cv.imread(img1, 0)
    img2 = cv.imread(img2, 0)
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches


# CREATING MATCH IMAGE USING RAINIER1 AND RAINIER2 IMAGE
img1 = cv.imread(img_rainier1, 0)
img2 = cv.imread(img_rainier2, 0)
kp1, kp2, matches = kp_matcher(img_rainier1, img_rainier2)
img_matches = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
cv.imwrite('2.png', img_matches)
cv.imshow('2.png', img_matches)
cv.waitKey(0)


# END OF STEP 2

# SIDE FUNCTION TO GET CODINATES FROM MATCHES
# takes in matches and key points from each image and outputs 2 lists of coordinates from each matches (x,y)
def potential_matches(matches, kp1, kp2):
    kp1_matches = []
    kp2_matches = []
    for m in matches:
        # img2
        trainIdx = m.trainIdx
        # img1
        queryIdx = m.queryIdx
        (x1, y1) = kp1[queryIdx].pt
        (x2, y2) = kp2[trainIdx].pt
        kp1_matches.append((x1, y1))
        kp2_matches.append((x2, y2))
    kp1_matches = np.float32(kp1_matches)
    kp2_matches = np.float32(kp2_matches)
    return kp1_matches, kp2_matches


# STEP 3 COMPUTING HOMOGRAPHY USING RANSAC

# 3-A
def project(x1, y1, H):
    x2 = (H[0][0] * x1 + H[0][1] * y1 + H[0][2]) / (H[2][0] * x1 + H[2][1] * y1 + H[2][2])
    y2 = (H[1][0] * x1 + H[1][1] * y1 + H[1][2]) / (H[2][0] * x1 + H[2][1] * y1 + H[2][2])
    return (x2, y2)


# 3-B
def computeInlierCount(H, kp1, kp2, inlierThreshold):
    num_matches = 0
    match = []
    for x in range(len(kp1)):
        proj = project(kp1[x][0], kp1[x][1], H)
        ssd = ((proj[0] - kp2[x][0]) ** 2 + (proj[1] - kp2[x][1]) ** 2)
        if ssd < inlierThreshold:
            num_matches += 1
            match.append(x)
    return num_matches, match


# 3-C
# takes in a list of matching kp1 to kp2
def RANSAC(kp1_matches, kp2_matches, numIterations, inlierThreshold):
    best_matches = 0
    match_positions = []
    # sampling random points for best homography

    for i in range(numIterations):
        src_coordinates = []
        des_coordinates = []
        for x in range(4):
            rand = random.randrange(len(kp1_matches))
            src_coordinates.append((kp1_matches[rand][0], kp1_matches[rand][1]))
            des_coordinates.append((kp2_matches[rand][0], kp2_matches[rand][1]))
        src_coordinates = np.array(src_coordinates)
        des_coordinates = np.array(des_coordinates)
        H, _ = cv.findHomography(src_coordinates, des_coordinates, 0)
        num_matches, matches = computeInlierCount(H, kp1_matches, kp2_matches, inlierThreshold)
        if num_matches > best_matches:
            best_matches = num_matches
            match_positions = matches

    # computing a new refined homography
    final_match1 = []
    final_match2 = []
    for x in match_positions:
        final_match1.append((kp1_matches[x][0], kp1_matches[x][1]))
        final_match2.append((kp2_matches[x][0], kp2_matches[x][1]))
    final_match1 = np.float32(final_match1)
    final_match2 = np.float32(final_match2)
    hom, _ = cv.findHomography(final_match1, final_match2, 0)
    homInv, _ = cv.findHomography(final_match2, final_match1, 0)

    # computing new matches using refined homography
    num_matches, matches = computeInlierCount(hom, kp1_matches, kp2_matches, inlierThreshold)
    match_positions_final = matches

    # displaying new matches using refined homography
    matchess = []
    kp11 = []
    kp21 = []
    counter = 0
    for k in match_positions_final:
        kp11.append(cv.KeyPoint(kp1_matches[k][0], kp1_matches[k][1], 16))
        kp21.append(cv.KeyPoint(kp2_matches[k][0], kp2_matches[k][1], 16))
        matchess.append(cv.DMatch(counter, counter, 0))
        counter += 1
    img1 = cv.imread(img_rainier1, 0)
    img2 = cv.imread(img_rainier2, 0)
    final = cv.drawMatches(img1, kp11, img2, kp21, matchess, None)
    cv.imshow('3.png', final)
    cv.waitKey(0)
    cv.imwrite('3.png', final)

    return hom, homInv

# END OF STEP 3

# STEP 4

def stitch(img1, img2, hom, invHom):
    img1 = cv.imread(img1)
    img2 = cv.imread(img2)

    # computing size of stiched image (x,y)
    top_left = project(0, 0, invHom)
    top_right = project(img2.shape[1], 0, invHom)
    bottom_left = project(0, img2.shape[0], invHom)
    bottom_right = project(img2.shape[1], img2.shape[0], invHom)
    max_w = int(max(top_right[0], bottom_right[0]))
    min_top = int(abs(min(top_left[1], top_right[1], 0)))
    img1_start = int(min_top)
    max_bot = max(bottom_right[1], bottom_left[1], img1.shape[0])
    max_h = int(min_top + max_bot)
    stiched_img = np.zeros((max_h + 1, max_w + 1, 3), np.uint8)
    stiched_img[img1_start:img1.shape[0] + img1_start, 0:img1.shape[1]] = img1

    # merging both images together and blending the joint
    alpha = 0.9
    for y in range(stiched_img.shape[0]):
        for x in range(stiched_img.shape[1]):
            pro = project(x, y - img1_start, hom)
            if pro[0] - 2 < 0 and pro[0] + 2 > 0 and 0 < pro[1] < img2.shape[0]:
                stiched_img[y][x] = stiched_img[y][x] * (alpha) + cv.getRectSubPix(img2, (1, 1), pro) * (1 - alpha)
            elif 0 < pro[0] < img2.shape[1] and 0 < pro[1] < img2.shape[0]:
                stiched_img[y][x] = cv.getRectSubPix(img2, (1, 1), pro)

    cv.imshow('stiched', stiched_img)
    cv.imwrite('4.png', stiched_img)
    cv.waitKey(0)

    return stiched_img


# END OF STEP 4

kp1, kp2, matches = kp_matcher(img_rainier1, img_rainier2)
kp1_matches, kp2_matches = potential_matches(matches, kp1, kp2)
hom, invHom = RANSAC(kp1_matches, kp2_matches, 200, 10)
stitch(img_rainier1, img_rainier2, hom, invHom)
