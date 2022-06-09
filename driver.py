import cv2
import timeit
import numpy as np
import features
import imutils

def cam(template):
    # video_src = 0
    # cam = cv2.VideoCapture(video_src)
    # cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # # get train features
    # # template = cv2.imread('logo_train.png')
    # template = cv2.imread(template)
    # train_features = features.getFeatures(template)
    # cur_time = timeit.default_timer()
    # frame_number = 0
    # scan_fps = 0
    # while True:
    #     frame_got, frame = cam.read()
    #     if frame_got is False:
    #         break

    #     frame_number += 1
    #     if not frame_number % 100:
    #         scan_fps = 1 / ((timeit.default_timer() - cur_time) / 100)
    #         cur_time = timeit.default_timer()

    #     region = features.detectFeatures(frame, train_features)

    #     cv2.putText(frame, f'FPS {scan_fps:.3f}', org=(0, 50),
    #                 fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #                 fontScale=1, color=(0, 0, 255))

    #     if region is not None:
    #         box = cv2.boxPoints(region)
    #         box = np.int0(box)
    #         cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    #     cv2.imshow("Preview", frame)
    #     if cv2.waitKey(10) == 27:
    #         break
    
    video_src = 0
    cam = cv2.VideoCapture(video_src)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # get train features
    # template = cv2.imread('logo_train.png')
    template = cv2.imread(template)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    cv2.imshow("Template", template)
    cur_time = timeit.default_timer()
    frame_number = 0
    scan_fps = 0
    while True:
        frame_got, frame = cam.read()
        if frame_got is False:
            break

        frame_number += 1
        if not frame_number % 100:
            scan_fps = 1 / ((timeit.default_timer() - cur_time) / 100)
            cur_time = timeit.default_timer()
        cv2.putText(frame, f'FPS {scan_fps:.3f}', org=(0, 50),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(0, 0, 255))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = None
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
        # draw a bounding box around the detected result and display the image
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # accuracy test (might not be implemented)
        cv2.putText(frame, f'accuracy: {maxVal/500000:.4f}%', org=(250, 50),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1, color=(0, 0, 255))
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            break
    # accuracy test (might not be implemented)
    print(image, maxVal/500000,"%")

def img(template, image):
    viz = False
    template = cv2.imread(template)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    cv2.imshow("Template", template)
    frame = cv2.imread(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = None
    while True:
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

<<<<<<< HEAD
        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # check to see if the iteration should be visualized
        if viz:
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                            (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)
        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (maxVal, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    # draw a bounding box around the detected result and display the image
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
=======
            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # check to see if the iteration should be visualized
            if viz:
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                                (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (maxVal, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # draw a bounding box around the detected result and display the image
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.imshow("Image", frame)
        if cv2.waitKey(10) == 27:
            break
>>>>>>> 84164903deddf8dc54b517dfe21163a1b213ff63
    # accuracy test (might not be implemented)
    print(image, maxVal/500000,"%")
        
    # get train features
    # template = cv2.imread('logo_train.png')
    # template = cv2.imread(template)
    # train_features = features.getFeatures(template)
    # image = cv2.imread(image)
    # region = features.detectFeatures(image, train_features)
    
    # if region is not None:
    #     box = cv2.boxPoints(region)
    #     box = np.int0(box)
    #     cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # cv2.imshow("Preview", image)
    # cv2.waitKey(0)