from GetElementBox import GetElementBox
from Detection import YOLODetection 
import cv2

if __name__ == "__main__":
    img_path = "/home/sangnt52/sangai/ZiMonster/raw_data/Image-007.bmp"
    
    get_boxes = GetElementBox(img_path) 
    boxes = get_boxes.run()
    
    detection = YOLODetection()
    res = detection.inference(image_inputs=boxes, batch=16, conf=0.8)
    vis_img = detection.visualize_defects(*res)
    cv2.imwrite("result.png", vis_img)