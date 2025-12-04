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

trained model on roboflow, downloaded weights to models/hecarim.pt. inference working - blue/gromp 97-99%, player 85-97%, mm_player 78-91%.

## bot.py - main jungle bot

state machine controller for automated jungle clearing:
- States: STARTUP_LEVELUP, STARTUP_BUY, WALKING_TO_CAMP, WAITING_FOR_SPAWN, ENGAGING, ATTACKING, CAMP_CLEARED, IDLE
- YOLO detection for camps and player
- OCR (pytesseract) for health bar reading with fallback logic
- cliclick for mouse/keyboard automation on macOS
- Grok AI runs in background thread for camp clear verification

clear order: blue_buff -> gromp -> wolves -> raptors -> red_buff -> krugs

camp HP values:
- blue_buff: 2300, gromp: 2050, wolves: 1600, raptors: 1200, red_buff: 2300, krugs: 1400

level up order:
- 0: Q (start), 1: W (after blue), 2: Q, 3: E, 4: Q, 5: W, 6: Q

abilities:
- Q spammed every 2 sec during combat
- W used when approaching camps
- Smite (F) spammed at blue_buff and krugs

## strategist.py - Grok AI integration

streaming responses with chain-of-thought visible in overlay. context-aware:
- zone probabilities based on minimap player distance
- map side (TOP_SIDE: blue/gromp/wolves, BOT_SIDE: raptors/red/krugs)
- game time awareness - before 1:30 tells Grok to "plan clear" instead of verify

only job: decide if camp is DEAD and should move to next camp

minimum game times before camp can be cleared:
- blue_buff: 1:38, gromp: 1:44, wolves: 1:50, raptors: 1:58, red_buff: 2:06, krugs: 2:16

## overlay.py - PyQt6 always-on-top overlays

thread-safe design (worker thread sets state dict, main thread reads and updates UI):
- BOT overlay: status logs (x=50, y=610, 240x130)
- GROK overlay: streaming thoughts + results (x=50, y=745, 240x200)

features:
- animated "Thinking..." with dots during Grok calls
- streaming text display shows chain-of-thought as it comes in
- final result replaces thinking animation
- 1 second delay between Grok calls

## som.py - Set-of-Mark regions

minimap positions for walking to camps:
- mm_blue (1440, 838), mm_gromp (1408, 829), mm_wolves (1435, 875)
- mm_raptors (1511, 905), mm_red (1531, 938), mm_krugs (1544, 965)

## key changes from session

1. added smite spam at blue_buff and krugs
2. added hardcoded max HP values for camps (OCR fallback)
3. made OCR never return None during combat (estimates HP decay)
4. added Grok for camp clear verification (not tips/analysis)
5. zone probability system for Grok context
6. map side detection (TOP_SIDE vs BOT_SIDE)
7. removed kiting entirely - just attack until Grok confirms dead
8. streaming Grok responses with chain-of-thought
9. overlay redesign - taller/narrower, fixed header with animated thinking
10. clicks relative to detected player position

## running

```bash
uv run python bot.py
```

requires XAI_API_KEY env var for Grok API.
