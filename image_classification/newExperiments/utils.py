import os 
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def read_file(path):
    if os.path.isfile(path):
        with open(path) as json_file:
            data = json.load(json_file)
        return data
    else:
        raise Exception("file doesn't exist: ", path)

def read_file_in_dir(root_dir, file_name):
    path = os.path.join(root_dir, file_name)
    return read_file(path)

def log_to_file(path, log_str):
#     if os.path.exists(path):
    with open(path, "a") as f:
        f.write(log_str + "\n")
#     else:
#         with open(path, "w") as f:
#             f.write(log_str + "\n")


def log_to_file_in_dir(root_dir, file_name, log_str):
    path = os.path.join(root_dir, file_name)
    log_to_file(path, log_str)
    
def write_to_file(path, data):
    with open(path, "w") as outfile:
        json.dump(data, outfile)


def write_to_file_in_dir(root_dir, file_name, data):
    path = os.path.join(root_dir, file_name)
    write_to_file(path, data)

            
def analyze_predictions(filename):
    with open(filename, 'r') as f:
        predictions = json.load(f)
    true_labels = predictions["true labels"]
    pred_labels = predictions["predicted labels"]
    cf_matrix =confusion_matrix(true_labels, pred_labels)
    print("printing confusion matrix")
    sns.heatmap(cf_matrix, annot=True)
    plt.show()