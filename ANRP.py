import numpy as np
import matplotlib.pyplot as plt
import cv2
import easyocr
import imutils


img = cv2.imread('car6.jpg')
ngray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(ngray, cv2.COLOR_BGR2RGB))
# plt.show()

bfilter = cv2.bilateralFilter(ngray, 11, 17, 17)
edge = cv2.Canny(bfilter,30,200)
# plt.imshow(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))
# plt.show()

points = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(points)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
location = None
for contour in contours:
    appr = cv2.approxPolyDP(contour, 10, True)
    if len(appr) == 4:
        location = appr
        break

# location

filtre = np.zeros(ngray.shape, np.uint8)
nimg = cv2.drawContours(filtre, [location], 0, 255, -1)
nimg = cv2.bitwise_and(ngray, filtre)
# plt.imshow(cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB))
# plt.show()

(x,y) = np.where(filtre == 255)
(i,j) = (np.min(x), np.min(y))
(k,l) = (np.max(x), np.max(y))
crimg = ngray[i:k+1, j:l+1]
# plt.imshow(cv2.cvtColor(crimg, cv2.COLOR_BGR2RGB))
# plt.show()

read = easyocr.Reader(['en', 'ar'])
result = read.readtext(crimg)

plat = result[0][-2]
print(plat)
