# Quick and dirty utility to get coordinates for transforming view into 
# a bird's eye view. Useful in OCRs were the camera is in a fixed positioning
# viewing a straight plane.  

import cv2
import numpy as np

def onTrackbarChange(trackbarValue):
    pass

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

def expandPerspective(rect ,  width, height):
	'''Expand the perspective out to the image limits
	by finding intersection using point-slope form'''
	# Constants
	x = 0
	y = 1
	
	# Convert coordinate system
	rect[:,1] *= -1
	(tl, tr, br, bl) = rect

	# Find the slope of each of the 4 lines
	slopeTop = (tr[y]-tl[y]) / (tr[x]-tl[x])
	slopeBottom = (br[y]-bl[y]) / (br[x]-bl[x])
	slopeLeft = (tl[y]-bl[y]) / (tl[x]-bl[x])
	slopeRight = (tr[y]-br[y]) / (tr[x]-br[x])

	# Assign new points based on image size
	pointRight = width,0
	pointTop = 0,0
	pointBottom = width, height * -1.0
	pointLeft = 0, height* -1.0

	# Find where the new expanded lines intersect using point slope form
	def intersectoin (m1,m2,x1,x2,y1,y2,orig):

		x = ((m2*x2-m1*x1)-(y2-y1))/(m2-m1)
		#y = ((-1.0*m1*y2 + m1*m2*x2 + y1*m2 )-(m1*m2*x1))/(m2-m1)
		y = m1*(x - x1) + y1

		try:
		   x = round(x)
		   y = round(y)
		except:
			return orig
		return x, y

	new_tr =  intersectoin (slopeTop,slopeRight,pointTop[x],pointRight[x],pointTop[y],pointRight[y],tr)
	new_tl =  intersectoin (slopeTop,slopeLeft,pointTop[x],pointLeft[x],pointTop[y],pointLeft[y],tl)
	new_br =  intersectoin (slopeBottom,slopeRight,pointBottom[x],pointRight[x],pointBottom[y],pointRight[y],br)
	new_bl =  intersectoin (slopeBottom,slopeLeft,pointBottom[x],pointLeft[x],pointBottom[y],pointLeft[y],bl)

	# Convert coordinate system back
	new_rect = rect = np.array([new_tl, new_tr, new_br, new_bl], dtype = "float32")
	new_rect[:,1] *= -1


	return new_rect

# Derived from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example

def four_point_transform(image, pts):
	# Unpack points
	rect = pts
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
		[0, 0], #tl
		[maxWidth - 1, 0], #tr
		[maxWidth - 1, maxHeight - 1], #br
		[0, maxHeight - 1]], #bl
		dtype = "float32")

	# Move image to positive coordinates
	min_x =  round(abs(np.min(rect[:,0])))
	min_y = round(abs(np.min(rect[:,1])))
	T = np.matrix( [[ 1 , 0 , min_x], # Get min x
					[ 0 , 1 , min_y ], # Get min y
					[ 0 , 0 ,    1    ]],
					dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, T * M  , (maxWidth + min_x , maxHeight + min_y), borderMode=cv2.BORDER_TRANSPARENT)
	# return the warped image
	return warped

# Open Image
img = cv2.imread('img\\example1.jpeg')

# Open windows for control, original image, and result
cv2.namedWindow('Control', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Main', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Birds Eye', cv2.WINDOW_AUTOSIZE)

# Track bars for coordinates
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
    warped = four_point_transform(img, expandPerspective(order_points(pts), img.shape[1], img.shape[0]))
    cv2.imshow('Birds Eye',warped)

	# Exit
    if cv2.waitKey(1)==27:
        exit(0)

cv.detroyAllWindows()

