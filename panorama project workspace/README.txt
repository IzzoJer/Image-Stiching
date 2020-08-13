Jeremie Izzo
40016103
Comp 425 - Project 

My code is split in section chronological to the steps in the assignment description. I commented the
 start and end of each section to keep things organized and easy to understand. The read me will go 
through my code step by step explaining each function and its purpose.

def detect_corners(src_img, out_name):

This function implements part one. This detects corners and outputs the image to display. This function 
uses sifts build in function. Takes in a source image and an output name to display the key points on

def kp_matcher(img1, img2):

This function is used to implement part 2. This matches key points of one image to another using sifts 
build in functions. It takes in the first and second image, computes the matches and returns key point 
of image one and two and returns the matches between them  

def potential_matches(matches, kp1, kp2):

This function is a helper function that outputs only the key points that match. This function takes in
the matches with key points from both images and creates 2 new arrays with only the key points that 
represent the best matches in parallel and returns them 

def project(x1, y1, H):

This function is used to project a point to a plane using the homography. It takes in an x and y value 
with the homography, computes the new coordinate and returns it.

def computeInlierCount(H, kp1, kp2, inlierThreshold):

 This function is used to count the number of inliers between a certain threshold. This function takes 
in a homograpy, list of key point from image one and 2 and takes in a threshold. This function computes 
the projection of the points in image 1 using the project functions and calculates the SSD of the projected 
point to the coordinates in image 2. If it is in the threshold then we add to the count. If not we disregard 
it. this function also keeps track of the matches saving the coordinates in an array. Returns the number of 
matches and the match positions.

def RANSAC(kp1_matches, kp2_matches, numIterations, inlierThreshold):
  
This function takes in key point matches from image one and 2, number of iterations and a threshold. The 
function starts by computing the best homography using 4 random points iterating the number of iterations set. 
Then once this holography is found I then use the list of matches to create a new refined homography using all 
the points found. I then recomputed the projection of key points using the new refined homography and display 
the result. The function returns the refined homography and inverse homography

def stitch(img1, img2, hom, invHom):

 this function takes in 2 images with a homograpgy and inverse homography. Uses this information to stich both 
images together. I start off by computing the 4 corners of the second image inverse projected and compute the 
size of the new stitched image. I insert the first image in the correct location on the stitched image. I then 
loop through each pixel of the stitched image and project the coordinate, if that coordinate lies within the 
boundaries of the 2nd image then I copy the value of the pixel to the stitched image. If the value is between a 
range of the border of the image then I use alpha blending to blend the edges.    

