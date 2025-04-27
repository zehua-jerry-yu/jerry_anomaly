import json
import os


def generate_fold_yaml_and_txt(fold_num, train_json, val_json, image_dir, output_dir):
    # Load the JSON files
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    
    with open(val_json, 'r') as f:
        val_data = json.load(f)

    # Extract category names (class names) from the categories field in train_data
    class_names = [category['name'] for category in train_data['categories']]
    num_classes = len(class_names)

    # Create a dictionary to map category names to class ids
    category_id_to_name = {category['id']: category['name'] for category in train_data['categories']}
    
    # Prepare the paths for train and val images
    train_image_paths = [os.path.join(image_dir, img['file_name']) for img in train_data['images']]
    val_image_paths = [os.path.join(image_dir, img['file_name']) for img in val_data['images']]

    # Function to convert COCO bbox to YOLO format [class_id, x_center, y_center, width, height]
    def convert_bbox_to_yolo_format(bbox, image_width, image_height):
        # COCO format: [x_min, y_min, width, height]
        x_min, y_min, width, height = bbox
        # YOLO format: [class_id, x_center, y_center, width, height]
        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        width = width / image_width
        height = height / image_height
        return [x_center, y_center, width, height]

    # Write the train.txt and val.txt files and their corresponding label files
    for image_data, data_type in zip([train_data, val_data], ['train', 'val']):
        # Open the txt files in write mode, so they are overwritten each time
        with open(os.path.join(output_dir, f'fold{fold_num}_{data_type}.txt'), 'w') as txt_file:
            for image_info in image_data['images']:
                image_path = os.path.join(image_dir, image_info['file_name'])
                image_id = image_info['id']
                image_width = image_info['width']
                image_height = image_info['height']
                
                # Prepare label file
                label_file_path = os.path.join(image_dir, f'{image_info["file_name"].split(".")[0]}.txt')
                os.makedirs(os.path.dirname(label_file_path), exist_ok=True)
                
                with open(label_file_path, 'w') as label_file:
                    # Get the annotations for this image
                    image_annotations = [anno for anno in image_data['annotations'] if anno['image_id'] == image_id]
                    
                    for anno in image_annotations:
                        class_id = anno['category_id'] - 1  # COCO category id starts from 1, YOLO starts from 0
                        bbox = anno['bbox']
                        # Convert the bbox to YOLO format
                        yolo_bbox = convert_bbox_to_yolo_format(bbox, image_width, image_height)
                        # Write to the label file
                        label_file.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
                
                # Add image path to train.txt or val.txt
                txt_file.write(image_path + '\n')

    # Write the foldi.yaml file
    yaml_content = f"""
train: {os.path.join(output_dir, f'fold{fold_num}_train.txt')}
val: {os.path.join(output_dir, f'fold{fold_num}_val.txt')}

nc: {num_classes}
names: {class_names}
"""
    with open(os.path.join(output_dir, f'fold{fold_num}.yaml'), 'w') as f:
        f.write(yaml_content)


if __name__ == "__main__":

    # Assuming JSON files are in the format train1.json, val1.json, ..., train5.json, val5.json
    image_dir = "/root/autodl-tmp/kaggle/anomaly/ssgd/SSGD/lb101"
    output_dir = "/root/autodl-tmp/kaggle/anomaly/ssgd/SSGD/annotations_lb101"
    folds = 5

    for i in range(1, folds + 1):
        train_json = f'{output_dir}/train{i}.json'
        val_json = f'{output_dir}/val{i}.json'
        generate_fold_yaml_and_txt(i, train_json, val_json, image_dir, output_dir)

    print("Files generated successfully!")