'''
The main script to be run if you want to do live face detection
Below I have incorporated a naive way of saving the image similar to what you did
However, I do not know whether png can support saving from a float64 array
When I opened the image it turned out to be black because of that
Because of this I would assume that the encoding into png will actually lower the number of
data points
I have tried to cast it into uint8 but a similar problem arose where the image becomes more obscured
Another naive solution could just be to omit the prewhiten step. 
Hoever I do not recommend that, apprently the accuracy is higher when it is kept, across images
of different luminance
The better way if you want to convert the data into storage for a separte recognition process is to
store it into a .npy file
This could be the solution to how you store multiple embedding 'keys' *hint hint*

Also with my gpu, i managed to get around 8.6 fps including time taken to get and write the image
but 9.8 fps without writing and getting the image
But its probbaly much worse with your laptop so you can try using the cascader instead.
But all things considered, MTCNN alot more consistent
'''

import cv2
import numpy as np
import os
import time

from detector import MTCNN

print('initialising detector')
detector = MTCNN()
cap = cv2.VideoCapture(0)

i=1
script_start = time.time()
process_time = 0
process_speed = 0

while cap.isOpened():
    ret, im_array = cap.read()
    print('reading')
    if not ret: break

    start_time = time.time()
    bbs = detector.get_faces(im_array)
    annotated = detector.draw_detections(im_array, bbs)
    crops = detector.extract_crops(im_array, bbs, True)

    # # if you want to write the crops, but it is a float64 so conversion to png is problematic
    # if not isinstance(crops, list):
    #     cv2.imwrite('./crops/image_{}.png'.format(i), crops)
    time_taken = time.time() - start_time
    process_time += time_taken

    if not i % 20:
        ave_fps = i / script_start
        process_speed = i / process_time
    
    annotated = cv2.putText(annotated, 'Process Speed: %.4d' % process_speed, (0, 20), \
        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    i += 1
    cv2.imshow('Live Detection', annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
# Speed stats
print('Average fps: ', ave_fps)
print('Average process speed: ', i/process_time)
print('No of frames taken over: ', i)