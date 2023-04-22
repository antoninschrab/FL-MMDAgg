# -------------------------------------------------
# PARAMETERS
# make sure those match the ones in shift_tester.py
# -------------------------------------------------

# old = True
# kernel_type = "laplace"
# kernel_type = "gaussian"

old = False
kernel_type = "laplace"
# kernel_type = "gaussian"
# kernel_type = "laplace_gaussian"
# kernel_type = "all"

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

import numpy as np
from tensorflow import set_random_seed
seed = 1
np.random.seed(seed)
set_random_seed(seed)

import keras
import tempfile
import keras.models

from keras import backend as K 
from shift_detector import *
from shift_locator import *
from shift_applicator import *
from data_utils import *
from shared_utils import *
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

datset = sys.argv[1]
test_type = sys.argv[3] 

# Define results path and create directory.
old_str = "_old" if old else "" 
path = './paper_results/'
path += test_type + old_str + '_' + kernel_type + '/'
path += datset + '_'
path += sys.argv[2] + '/'
if not os.path.exists(path):
    os.makedirs(path)

# Define DR methods.
dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value, DimensionalityReduction.BBSDh.value]
if test_type in ['multiv', 'mmdagg']:
    #dr_techniques = [DimensionalityReduction.NoRed.value, DimensionalityReduction.PCA.value, DimensionalityReduction.SRP.value, DimensionalityReduction.UAE.value, DimensionalityReduction.TAE.value, DimensionalityReduction.BBSDs.value]
    #dr_techniques_str = ["NoRed", "PCA", "SRP", "UAE", "TAE", "BBSDs"]
    dr_techniques = [DimensionalityReduction.NoRed.value, ]
    dr_techniques_str = ["NoRed", ]
if test_type == 'univ':
    dr_techniques_plot = dr_techniques.copy()
    dr_techniques_plot.append(DimensionalityReduction.Classif.value)
else:
    dr_techniques_plot = dr_techniques.copy()

# Define test types and general test sample sizes.
test_types = [td.value for td in TestDimensionality]
if test_type in ['multiv', 'mmdagg']:
    od_tests = []
    if test_type == 'multiv':
        md_tests = [MultidimensionalTest.MMD.value]
    if test_type == 'mmdagg':
        md_tests = [MultidimensionalTest.MMDAgg.value]
    samples = [10, 20, 50, 100, 200, 500, 1000, 10000] 
else:
    od_tests = [OnedimensionalTest.KS.value]
    md_tests = []
    samples = [10, 20, 50, 100, 200, 500, 1000, 10000]
difference_samples = 20

# Number of random runs to average results over.
random_runs = 5

# Significance level.
sign_level = 0.05

# Whether to calculate accuracy for malignancy quantification.
calc_acc = True

# Define shift types.
if sys.argv[2] == 'small_gn_shift':
    shifts = ['small_gn_shift_0.1',
              'small_gn_shift_0.5',
              'small_gn_shift_1.0']
elif sys.argv[2] == 'medium_gn_shift':
    shifts = ['medium_gn_shift_0.1',
              'medium_gn_shift_0.5',
              'medium_gn_shift_1.0']
elif sys.argv[2] == 'large_gn_shift':
    shifts = ['large_gn_shift_0.1',
              'large_gn_shift_0.5',
              'large_gn_shift_1.0']
elif sys.argv[2] == 'adversarial_shift':
    shifts = ['adversarial_shift_0.1',
              'adversarial_shift_0.5',
              'adversarial_shift_1.0']
elif sys.argv[2] == 'ko_shift':
    shifts = ['ko_shift_0.1',
              'ko_shift_0.5',
              'ko_shift_1.0']
elif sys.argv[2] == 'orig':
    shifts = ['rand', 'orig']
    brightness = [1.25, 0.75]
elif sys.argv[2] == 'small_image_shift':
    shifts = ['small_img_shift_0.1',
              'small_img_shift_0.5',
              'small_img_shift_1.0']
elif sys.argv[2] == 'medium_image_shift':
    shifts = ['medium_img_shift_0.1',
              'medium_img_shift_0.5',
              'medium_img_shift_1.0']
elif sys.argv[2] == 'large_image_shift':
    shifts = ['large_img_shift_0.1',
              'large_img_shift_0.5',
              'large_img_shift_1.0']
elif sys.argv[2] == 'medium_img_shift+ko_shift':
    shifts = ['medium_img_shift_0.5+ko_shift_0.1',
              'medium_img_shift_0.5+ko_shift_0.5',
              'medium_img_shift_0.5+ko_shift_1.0']
elif sys.argv[2] == 'only_zero_shift+medium_img_shift':
    shifts = ['only_zero_shift+medium_img_shift_0.1',
              'only_zero_shift+medium_img_shift_0.5',
              'only_zero_shift+medium_img_shift_1.0']
else:
    shifts = []

# -------------------------------------------------
# PIPELINE START
# -------------------------------------------------

# Stores p-values for all experiments of a shift class.
samples_shifts_rands_dr_tech = np.ones((len(samples), len(shifts), random_runs, len(dr_techniques_plot))) * (-1)

red_dim = -1
red_models = [None] * len(DimensionalityReduction)

samples_backup = samples.copy()

# Iterate over all shifts in a shift class.
for shift_idx, shift in enumerate(shifts):

    samples = samples_backup.copy()

    shift_path = path + shift + '/'
    if not os.path.exists(shift_path):
        os.makedirs(shift_path)

    # Stores p-values for a single shift.
    rand_run_p_vals = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)

    # Stores shift decisions for a single shift.
    rand_run_decs = np.ones((len(samples), len(dr_techniques_plot), random_runs)) * (-1)

    # Stores accuracy values for malignancy detection.
    val_accs = np.ones((random_runs, len(samples))) * (-1)
    te_accs = np.ones((random_runs, len(samples))) * (-1)
    dcl_accs = np.ones((len(samples), random_runs)) * (-1)

    samples_copy = samples.copy()
    for sample in samples_copy:
        new_path = shift_path + str(sample) + '/'
        if os.path.exists(new_path):
            samples.remove(sample)
    if samples == []:
       print("All test outputs for sample sizes", samples_copy, "have already been computed for", datset, shift, ".")
    else:
        print("Computing test outputs for sample sizes", samples, "for", datset, shift, ".")
        print("Test outputs for other sample sizes", [s for s in samples_copy if s not in samples], "have already been computed for", datset, shift, ".")
        # Average over a few random runs to quantify robustness.
        #for rand_run in range(random_runs):
        for rand_run in range(0, random_runs-1):
            rand_run = int(rand_run)
    
            print("Random run %s" % rand_run)
    
            np.random.seed(rand_run)
            set_random_seed(rand_run)
    
            # Load data.
            (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = \
                import_dataset(datset, shuffle=True)
            X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
            X_te_orig = normalize_datapoints(X_te_orig, 255.)
            X_val_orig = normalize_datapoints(X_val_orig, 255.)
    
            # Apply shift.
            if shift == 'orig':
                print('Original')
                (X_tr_orig, y_tr_orig), (X_val_orig, y_val_orig), (X_te_orig, y_te_orig), orig_dims, nb_classes = import_dataset(datset)
                X_tr_orig = normalize_datapoints(X_tr_orig, 255.)
                X_te_orig = normalize_datapoints(X_te_orig, 255.)
                X_val_orig = normalize_datapoints(X_val_orig, 255.)
                X_te_1 = X_te_orig.copy()
                y_te_1 = y_te_orig.copy()
            else:
                (X_te_1, y_te_1) = apply_shift(X_te_orig, y_te_orig, shift, orig_dims, datset)
    
            X_te_2 , y_te_2 = random_shuffle(X_te_1, y_te_1)
    
            # Check detection performance for different numbers of samples from test.
            for si, sample in enumerate(samples):
    
                print("Sample %s" % sample)
    
                X_te_3 = X_te_2[:sample,:]
                y_te_3 = y_te_2[:sample]
    
                X_val_3 = X_val_orig[:sample,:]
                y_val_3 = y_val_orig[:sample]
    
                X_tr_3 = np.copy(X_tr_orig)
                y_tr_3 = np.copy(y_tr_orig)
    
                # Detect shift.
                shift_detector = ShiftDetector(dr_techniques, test_types, od_tests, md_tests, sign_level, red_models,
                                               sample, datset)
                (od_decs, ind_od_decs, ind_od_p_vals), \
                (md_decs, ind_md_decs, ind_md_p_vals), \
                red_dim, red_models, val_acc, te_acc = shift_detector.detect_data_shift(X_tr_3, y_tr_3, X_val_3, y_val_3,
                                                                                        X_te_3, y_te_3, orig_dims,
                                                                                        nb_classes)
                
                val_accs[rand_run, si] = val_acc
                te_accs[rand_run, si] = te_acc
    
                if test_type in ['multiv', 'mmdagg']:
                    print("Shift decision: ", ind_md_decs.flatten())
                    print("Shift p-vals: ", ind_md_p_vals.flatten())
    
                    rand_run_decs[si,:,rand_run] = ind_md_decs.flatten()
                    rand_run_p_vals[si,:,rand_run] = ind_md_p_vals.flatten()
                else:
                    raise ValueError("Only test_type 'multiv', 'mmdagg' is implemented. ")
    
            for sample_idx, sample in enumerate(samples):
                for dr_idx, dr in enumerate(dr_techniques_str):
                    new_path = shift_path + str(sample) + '/' + dr + '/'
                    if not os.path.exists(new_path):
                        os.makedirs(new_path)
                    new_decs = rand_run_decs[sample_idx, dr_idx]
                    np.savetxt("%s/mean_decs.csv" % new_path, new_decs, delimiter=",") 
