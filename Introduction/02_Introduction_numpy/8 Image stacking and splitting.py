import numpy as np

a = np.arange(27)
print a
# print type(a)

# a.reshape(3, 9)
# print a
a2 = a.reshape(3, 9)
print a2

a3 = a.reshape(3, 3, 3)
print a3

b = [[1, 2], [3, 4]]
print b
print type(b)

b2 = np.array(b)
print b2
print type(b2)

import cv2

image_grey = cv2.imread('smallgray.png', 0) # 0 - gray, 1 - BGR
print image_grey

# print np.min(np.array([[1,2],[3,4]]))
image_grey_new = image_grey - np.min(image_grey)
print image_grey_new
cv2.imwrite('smallgray_new.png', image_grey_new)

for i in image_grey:
    print i

for i in image_grey.T:
    print i

for i in image_grey.flat:
    print i

image_grey_hstack = np.hstack((image_grey,image_grey,image_grey,image_grey))
print image_grey_hstack
cv2.imwrite('smallgray_hstack.png', image_grey_hstack)

image_grey_vstack = np.vstack((image_grey,image_grey,image_grey,image_grey))
print image_grey_vstack
cv2.imwrite('smallgray_vstack.png', image_grey_vstack)

image_grey_hstack_hsplit = np.hsplit(image_grey_hstack,2)
print image_grey_hstack_hsplit
cv2.imwrite('smallgray_hstack_hsplit.png', image_grey_hstack_hsplit[1])

image_grey_vstack_vsplit = np.vsplit(image_grey_vstack,2)
print image_grey_vstack_vsplit
cv2.imwrite('smallgray_vstack_vsplit.png', image_grey_vstack_vsplit[1])




