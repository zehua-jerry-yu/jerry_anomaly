import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import os
from detectron2 import model_zoo


import json
path_root = "/root/autodl-tmp/kaggle/anomaly/ssgd/SSGD/annotations_lb101"
for i in (1, 2, 3, 4, 5):
    for type in ("train", "val"):
        with open(os.path.join(path_root, f"{type}{i}.json"), "r") as f:
            data = json.load(f)
        for ann in data["annotations"]:
            # detectron expects [[]] for segmentation, not []
            seg = ann.get("segmentation", [])
            ann["segmentation"] = [seg]  # wrap in a list
        with open(os.path.join(path_root, f"{type}{i}_detec.json"), "w") as f:
            json.dump(data, f)


# Step 1: Register your dataset (in COCO format)
register_coco_instances(
    "my_dataset_train", {}, 
    "/root/autodl-tmp/kaggle/anomaly/ssgd/SSGD/annotations_lb101/train1_detec.json", 
    "/root/autodl-tmp/kaggle/anomaly/ssgd/SSGD/lb101"
)
register_coco_instances(
    "my_dataset_val", {}, 
    "/root/autodl-tmp/kaggle/anomaly/ssgd/SSGD/annotations_lb101/val1_detec.json", 
    "/root/autodl-tmp/kaggle/anomaly/ssgd/SSGD/lb101"
)

# Step 2: Set up configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000  # Adjust depending on your dataset size
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(data["categories"])  # Change to your number of classes
cfg.OUTPUT_DIR = "/root/autodl-tmp/kaggle/anomaly/detectron_output"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Step 3: Train
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# evaluate model

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

evaluator = COCOEvaluator("my_dataset_val", output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "my_dataset_val")
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
print(metrics)