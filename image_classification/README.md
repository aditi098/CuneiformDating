To run ResNet code on the new split of data, go to newExperiments.
1. Make a new config under newExperiments/configs/. Make sure to update the experiment name as well.
2. Update main.py to load or test on an existing model or train a new model. Simply add ```exp.load(model_path)``` or ```exp.test()``` or ```exp.train()```.
3. Run ```python main.py experiment_name```
