- `/jupyterNotebooks` has some notebooks for quick testing and visualization
  - `analyzeSegmentation.ipynb` - has code for analyzing the quality of segmentation, not needed to run segmentation

- `/code` has scripts and utility functions to run segmentAnything and ensemble of three kinds of segmentation
  - `ensembleSegmentation.py` - combine segmentAnything, line based and rule based segmentation and get IoU, it saves the IoU information in `/code/temp_results/iou_segmentation.json`
  -  `ensembleSegmentationUtils.py` - has utility functions used in the above script
  -  `segmentAnything.py` - most updated script to run segmentAnything on cuneiform data
  -  `segmentAnythingUtils.py` - has utils for the above file
