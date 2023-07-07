# CuneiformDating

The folder data_split has the following files:
- createSplit.ipynb: contains code to create train, val, test split to ensure equal representation of all classes in val and test set.
- train_val_test_split.json



train_val_test_split.json has the new data split. It is created using the logic in createSplit.ipynb with batch_class_size = 300 and min_class_size = 100 
