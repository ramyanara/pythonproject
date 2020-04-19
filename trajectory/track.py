import numpy as np
import cv2
#import os
import imutils

cap = cv2.VideoCapture('overpass_Trim.wmv')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))
"""def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)"""
"""while True:
    rect, frame = cap.read()
    frame75 = rescale_frame(frame, percent=75)"""
    #cv2.imshow('frame75', frame75)
   # frame150 = rescale_frame(frame, percent=150)
   # cv2.imshow('frame150', frame150)

# Take first frame and find corners in it
ret, old_frame = cap.read()

#frame75 = rescale_frame(old_frame, percent=75)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(cap.get(prop))
	#print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video fwriter = Noneile
except:
	#print("[INFO] could not determine # of frames in video")
	#print("[INFO] no approx. completion time can be provided")
	total = -1


while(1):
    ret,frame = cap.read()
    #rescale_frame(frame, percent=75)
    #cv2.resize(frame, (1280, 720))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

        # initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*"WMV2")
    writer = cv2.VideoWriter("output/car.avi", fourcc, 30*total,
                                 (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)

cv2.destroyAllWindows()
cap.release()