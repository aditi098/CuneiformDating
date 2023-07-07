# CuneiformDating

The folder data_split has the following files:
- createSplit.ipynb: contains code to create train, val, test split to ensure equal representation of all classes in val and test set.
- train_val_test_split.json: contains the train, val, test split created using batch_class_size = 300 and min_class_size = 100 These two parameters can be changed to produce a new split.
- period_to_label_mapping.json: has mapping from label (i.e., 1,2,3,4, etc) to class names, i.e., Uruk IV, Uruk V etc
- train_ids_class_wise.json: This has a dictionary of lists, where each list contains train ids from one specific class. This is later used to create balanced dataloaders (to handle class imbalance).
 
The folder image_classification is divided into three parts:
- oldExperiments: This folder contains data, code and results for classical ML methods and ResNet on an older split of data (using only 1000 samples from every class)
- newExperiments: This folder contains code and results of ResNet experiments on the new split of data (from data_split/train_val_test_split.json)
- segmentation: This folder contains model and code for segmenting images using SegmentAnything and obtaining front faces

The data can be found on the baobab server under /trunk/shared/cuneiform/full_data. These haven't been uploaded to github because of size issues. 
- all_ids.json: contains pids of all tablets that have an image 
- expanded_catalogue.csv: contains metadata of all tablets scraped from CDLI. These might or might not have an image associated with them.
- images: This folder contains all the original tablet images scraped from CDLI.
- scripts: This folder has some basic scripts used to prepare data.
- segmented_images: This folder has segmented images, i.e, just the front face cutout from original images.
- segmented_mask_info_compressed: This folder has results of SegmentAnything for every image.
- small_images_no_mask.json: This file contains a list of images that are low resolution and have no mask info associated with them in segmented_mask_info_compressed. Since these images are low resolution, they weren't processed using SegmentAnything and are used directly without cutting out any faces.
