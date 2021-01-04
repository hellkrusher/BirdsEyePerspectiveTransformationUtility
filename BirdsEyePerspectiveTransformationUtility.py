# Quick and dirty utility to get coordinates for transforming view into 
# a bird's eye view. Useful in OCRs were the camera is in a fixed positioning
# viewing a straight plane.  

import cv2
import numpy as np

def onTrackbarChange(trackbarValue):
    pass

# From https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example
def order_points(pts):
	# initialize a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordinates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped

# Open Image
img = cv2.imread('img\\example1.jpeg')

# Open windows for control, original image, and result
cv2.namedWindow('Control', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Main', cv2.WINDOW_NORMAL)
cv2.namedWindow('Birds Eye', cv2.WINDOW_NORMAL)

# Track bars for concordance
cv2.createTrackbar( 'X L Bot', 'Control', 0, img.shape[1], onTrackbarChange )
cv2.createTrackbar( 'Y L Bot', 'Control', img.shape[0], img.shape[0], onTrackbarChange )

cv2.createTrackbar( 'X L Top', 'Control', 0, img.shape[1], onTrackbarChange )
cv2.createTrackbar( 'Y L Top', 'Control', 0, img.shape[0], onTrackbarChange )

cv2.createTrackbar( 'X R Top', 'Control', img.shape[1], img.shape[1], onTrackbarChange )
cv2.createTrackbar( 'Y R Top', 'Control', 0, img.shape[0], onTrackbarChange )

cv2.createTrackbar( 'X R Bot', 'Control', img.shape[1], img.shape[1], onTrackbarChange )
cv2.createTrackbar( 'Y R Bot', 'Control', img.shape[0], img.shape[0], onTrackbarChange )

# Loop
while(1):
	# Get Track Bar positions
    pts = np.array(eval('[(' + str(cv2.getTrackbarPos('X L Bot','Control')) + ',' + str(cv2.getTrackbarPos('Y L Bot','Control')) + '),' + 
                        '(' + str(cv2.getTrackbarPos('X L Top','Control')) + ',' + str(cv2.getTrackbarPos('Y L Top','Control'))+ '),' +
                        '(' + str(cv2.getTrackbarPos('X R Top','Control')) + ',' + str(cv2.getTrackbarPos('Y R Top','Control'))+ '),' +
                        '(' + str(cv2.getTrackbarPos('X R Bot','Control')) + ',' + str(cv2.getTrackbarPos('Y R Bot','Control'))+ ')]'
                        ), dtype = "int32")

	# Draw the perspective
    imgConnectedPoints = cv2.polylines(img.copy(), [pts], isClosed = True, color = (0,255,0), thickness = 3) 
    cv2.imshow('Main',imgConnectedPoints)

	# Draw the transformed bird's eye view
    warped = four_point_transform(img, pts)
    cv2.imshow('Birds Eye',warped)

	# Exit
    if cv2.waitKey(1)==27:
        exit(0)

cv.detroyAllWindows()

