import cv2
import numpy as np
import os

class GetElementBox:
    def __init__(self, img_path):
        self.img_path = img_path 
        
    def run(self):
        results =  {}
        img = cv2.imread(self.img_path,cv2.IMREAD_UNCHANGED)
        results["original_image"] = img
        blue_channel = img[:,:,0]

        #threshold inRange using cv2.inRange
        background_zone = cv2.inRange(blue_channel, 130,  255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(background_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour)<100:
                continue
            cv2.drawContours(background_zone, [contour], 0, 255, -1)

        removed_bg = cv2.bitwise_and(blue_channel,background_zone)
        selected_zone = cv2.inRange(removed_bg, 81, 174, cv2.THRESH_BINARY)
        # #perform closing morphology with circle kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        closing_selected_zone = cv2.morphologyEx(selected_zone, cv2.MORPH_CLOSE, kernel, None,None,None, 4)
        contours, _ = cv2.findContours(selected_zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        selected_zone_filtered = np.zeros_like(blue_channel)
        for contour in contours:
            if cv2.contourArea(contour)<10000:
                continue
            cv2.drawContours(selected_zone_filtered, [contour], 0, 255, -1)

        removed_unwanted = cv2.bitwise_and(removed_bg,selected_zone_filtered)
        rects_thresh = cv2.inRange(removed_unwanted,69,90,cv2.THRESH_BINARY)
        # # perform closing morphology with circle kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        closing = cv2.morphologyEx(rects_thresh, cv2.MORPH_CLOSE, kernel, None,None,None, 4)
        rects_filtered = np.zeros_like(blue_channel)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour)<1000:
                continue
            cv2.drawContours(rects_filtered, [contour], 0, 255, -1)

        # #find connected components and remove small area labels
        numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(rects_filtered,  8, cv2.CV_32S)

        list_of_boxes = []
        list_of_cropped_images = []
        for i in range(1, numLabels):
            #the first feature is always the background so we ignore it
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            list_of_boxes.append((x,y,w,h))

            crop_img = img[y:y+h, x:x+w]
            list_of_cropped_images.append(crop_img)
        results["image_batch"] = list_of_cropped_images
        results["boxes"] = list_of_boxes
        
        return results
    
if __name__ == "__main__":
    element_boxes = GetElementBox(img_path="/home/sangnt52/sangai/defect/raw_data/7.bmp")
    res = element_boxes.run() 
    print(list(res.keys()))