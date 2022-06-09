import cv2
import timeit
import numpy as np
import imutils

def detection(template,image=None):
    # set the camera capture window if image is none, implying they chose to run camera
    if image is None:
        video_src = 0
        cam = cv2.VideoCapture(video_src)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cur_time = timeit.default_timer()
        frame_number = 0
        scan_fps = 0
    
    # template = cv2.imread('logo_train.png')
    template = cv2.imread(template)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    cv2.imshow("Template", template)
    
    while True:
        # run camera if there is no image
        if image is None:
            frame_got, frame = cam.read()
            if frame_got is False:
                break
            frame_number += 1
            if not frame_number % 100:
                scan_fps = 1 / ((timeit.default_timer() - cur_time) / 100)
                cur_time = timeit.default_timer()
            cv2.putText(frame, f'FPS {scan_fps:.3f}', org=(0, 250),
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=0.5, color=(0, 0, 255))
        else:
            frame = cv2.imread(image)
        
        # change the frame into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found = None
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
            
        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (maxVal, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        
        # calculate the maximum correlation value by comparing template to itself
        maxAcc = cv2.matchTemplate(template, template, cv2.TM_CCOEFF)
        (_, maxAcc, _, _) = cv2.minMaxLoc(maxAcc)
        maxVal = maxVal/maxAcc
        # accuracy test (might not be implemented)
        cv2.putText(frame, f'accuracy: {maxVal:.4f}%', org=(0, 50),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(0, 0, 255))
        
        # if the accuracy is >15%, draw a bounding box around the detected result
        if maxVal >= 15:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        # show the frame
        cv2.imshow("Preview", frame)

        # break if Escape button is pressed or there is image (to avoid looping infinitely)
        if cv2.waitKey(10) == 27 or image is not None:
            break
        
    # accuracy test (for logging purposes)
    print(image, maxVal,"%")
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    print("Welcome to Logo Detection program with accuracy ********")
    while True:
        # loading template (maybe made it into an array with multiple logos?)
        template = 'logo_train.png'
        image = None
        choice = input("Enter '0' to upload image, otherwise it will run using camera. \
                        Press Enter to exit.\n>  ")
        # pressed Enter with no input, break loop
        if choice == '':
            break 
        # chose 0 for image upload
        if choice == '0':
            image = input("Enter the full path of the image (only .png): ")
            if cv2.imread(image) is None:
                image = 'test.apng'
        # run detection function, image will default to None 
        # to imply the user chose to run with the camera
        detection(template, image)

if __name__ == '__main__':
    main()