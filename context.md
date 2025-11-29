# gclear context log

set up uv project with python 3.12, added pyqt6/pillow/quartz deps. created overlay.py - transparent overlay showing roi boxes on screen with control panel for capturing screenshots. saves full + cropped rois to data/raw/<session>/. space to capture, previews update live.

moved overlay.py and config.py into data_collection/ folder since gclear will have other components later. updated data_dir path to still save to project root data/ folder.

fixed space key capture using qshortcut, fixed roi preview by properly setting bytes_per_line in qimage. removed "center" roi - full screenshot already captures entire game area.

extracted 83 frames from hecarim clear recording, trained yolo model on roboflow with 8 classes: player, mm_player, red_buff, blue_buff, gromp, wolves, raptors, krugs. added roboflow sdk and test_model.py for inference.

added ultralytics for local yolo training. train.py downloads dataset from roboflow and trains yolov8n on mps. training on roboflow instead for now.

cleaned up project: removed pycache, failed training runs, empty files. added .gitignore for python/ml/data files. created models/ folder for weights. moved api key to env var.

added calibrate.py - drag rois to reposition, drag corners to resize, define minimap zones by drawing. exports config for copy-paste into config.py.

added TRAINING.md with full instructions for training on linux gpu - dataset download, yolo commands, class info, project context.

created clean `context` branch with code only (no images/videos/datasets) for github push.
