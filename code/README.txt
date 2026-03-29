# A2-Classification
GEO5017 A2 - ReadMe
===================

Files
-----
This code folder contains the following Python files:
- main.py                : main entry point; runs the full pipeline
- features.py            : computes object features from the point clouds and writes data.txt
- feature_selection.py   : forward search using the scatter criterion
- classifiers.py         : SVM and Random Forest training/evaluation
- learning_curves.py     : manual learning-curve computation and plotting
- fixed_param.py         : optional helper for running fixed classifier settings

Entry point
-----------
Run only:
    python main.py

The code will:
1. compute features from the point clouds,
2. save/load the feature table in data.txt,
3. perform forward feature selection,
4. train and evaluate SVM and Random Forest,
5. generate learning-curve figures.

Required Python packages
------------------------
Install the required packages with:
    pip install numpy matplotlib scikit-learn scipy tqdm

Data path
---------
IMPORTANT: the path to the folder containing the 500 point clouds is set inside main.py.
Before running the code, check the variable called 'path' in main.py and make sure it points to your local point-cloud folder.

Recommended folder structure:
    project/  
      code/
        pointclouds-500/
        main.py
        features.py
        feature_selection.py
        classifiers.py
        learning_curves.py
    

If you want to avoid a machine-specific absolute path, a better option is to use a relative path in main.py.
For example, if 'code' and 'pointclouds-500' are sibling folders:

    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    path = BASE_DIR.parent / "pointclouds-500"

About data.txt
--------------
The feature extraction step writes the computed features to data.txt.
If data.txt already exists, the code may reuse it instead of recomputing features.
We have deleted the data.txt file, so that in your first run the features will be computed

Outputs
-------
The code prints the selected features, classifier performance, confusion matrices, and classification reports to the terminal.
It also saves the learning-curve figures in the code folder, for example:
- learning_curve_svm_error.png
- learning_curve_rf_error.png

Recommended file order in the submission
----------------------------------------
The code is designed to run from main.py only, so renaming is not strictly necessary.
However, if you want to make the order explicit in the archive, a clear naming scheme would be:
- 1_main.py
- 2_features.py
- 3_feature_selection.py
- 4_classifiers.py
- 5_learning_curves.py

If you keep the current file names, make sure main.py remains the main entry point.

How to reproduce the results
----------------------------
1. Place all .py files in the 'code' subfolder.
2. Make sure the point-cloud folder path in main.py is correct.
3. Install the required packages.
5. Run:
       python main.py