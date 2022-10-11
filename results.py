import numpy as np
import os
import glob

sample_sizes = [10, 20, 50, 100, 200, 500, 1000, 10000]
dr_techniques = ["NoRed", ] 

tests = [x[16:] for x in glob.glob("./paper_results/*")]
datasets = ["mnist", "cifar10"]
shift_types = [
    "adversarial_shift",
    "ko_shift",
    "large_gn_shift",
    "medium_gn_shift",
    "only_zero_shift+medium_img_shift",
    "small_gn_shift",
] 
deltas = ["_0.1", "_0.5", "_1.0"]

for test in tests:

    directories = ["paper_results/" + test + "/" + dataset + "_" + shift_type + "/" + shift_type + delta + "/" for dataset in datasets for shift_type in shift_types for delta in deltas]
    
    for dataset in datasets:
        for t in ("small", "medium", "large"):
            for delta in deltas:
                directories.append("paper_results/" + test + "/" + dataset + "_" + t + "_image_shift/" + t + "_img_shift" + delta + "/")
    
    for dataset in datasets:
        for delta in deltas:
            directories.append("paper_results/" + test + "/" + dataset + "_medium_img_shift+ko_shift/medium_img_shift_0.5+ko_shift" + delta + "/")
    
    filename = "mean_decs.csv"
    for sample_size in sample_sizes:
        if os.path.exists(directories[0] + str(sample_size) + "/"):
            print(" ")
            for dr_technique in dr_techniques: 
                power = []
                for directory in directories:
                    path = directory + str(sample_size) + "/" + dr_technique + "/" + filename
                    x = np.genfromtxt(path, delimiter=",").reshape((1, -1))
                    if np.min(x) < 0:
                        print("Negative valued encountered:", path)
                        print(x)
                        print(" ")
                    power.append(np.mean(x))
                print(test, sample_size, dr_technique, ":", np.round_(np.mean(power), 3))

