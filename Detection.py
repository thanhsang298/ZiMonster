from ultralytics import YOLO 
import cv2 
from PIL import Image 


class YOLODetection:
    def __init__(self, weight_path="./weight/best.pt"):
        self.weight_path = weight_path
        self.model = YOLO(self.weight_path)
        
    def mapping_defect_box_to_original_image(self, cropped_image_coords, defect_boxes):
        list_of_mapping_defect_boxes = []
        x_offset, y_offset, _, _ = cropped_image_coords
        for defect_box in defect_boxes:
            x, y, w, h = defect_box
            x_big = x + x_offset
            y_big = y + y_offset
            list_of_mapping_defect_boxes.append((x_big, y_big, w, h))
        return list_of_mapping_defect_boxes
    
    def visualize_defects(self, original_image, all_defects):
        for (x, y, w, h) in all_defects:
            top_left = (int(x)-5, int(y)-5)
            bottom_right = (int(x + w), int(y + h))
            cv2.rectangle(original_image, top_left, bottom_right, (0, 0, 255), 1)
        return original_image
    
    def inference(self, image_inputs, batch, conf=0.5):
        """
        - image_inputs: {
            "original_image": image_array,
            "cropped_images": list of cropped images,
            "boxes": list of boxes
        }
        """
        all_defects = []
        original_image = image_inputs["original_image"]
        cropped_images = image_inputs["image_batch"]
        boxes = image_inputs["boxes"]
        
        for i in range(0, len(cropped_images), batch):
            batch_images = cropped_images[i:i+batch]
            preds = self.model.predict(batch_images, conf)
            
            for batch_idx, pred in enumerate(preds):
                try:
                    defect_boxes = pred.boxes.xywh.cpu().numpy()
                    if len(defect_boxes):
                        cropped_image_coords = boxes[i + batch_idx]
                        mapping_coords = self.mapping_defect_box_to_original_image(cropped_image_coords, defect_boxes)
                        all_defects.extend(mapping_coords)
                except:
                    continue
        return (original_image, all_defects)