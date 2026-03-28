from features import feature_preparation
from classifiers import data_loading, SVM_classification, RF_classification, FEATURE_NAMES
from feature_selection import forward_search, print_forward_history
from learning_curves import compute_learning_curve, print_learning_curve_results, plot_learning_curve



INITIAL_FEATURES = [
    'height',
    'root_density',
    'area',
    'shape_index',
    'linearity',
    'sphericity',
    'slenderness',
    'length_height_ratio',
    'circularity',
    'footprint_density',
]

TRAIN_RATIOS = [i / 10.0 for i in range(1, 10)]  # 0.1, 0.2, ..., 0.9
N_REPEATS = 10

if __name__ == '__main__':
    path = "C:/Users/A3ano/OneDrive/Documenten/ML in the BE/Assingment 2/A2-Classification/pointclouds-500"

    print('Start preparing features')
    feature_preparation(data_path=path)

    print('Start loading data from the local file')
    ID, X, y = data_loading()

    print('\nInitial designed feature set:')
    print(INITIAL_FEATURES)

    selected_indices, selected_names, history = forward_search(
        X,
        y,
        FEATURE_NAMES,
        INITIAL_FEATURES,
        d=4,
        normalize=True,
    )

    print_forward_history(history)
    print('\nFinal selected 4 features from forward search:')
    print(selected_names)

    print('\nStart SVM classification with the 4 forward-selected features')
    svm_model_selected, svm_acc_selected, svm_best_params, svm_cv_score = SVM_classification(
        X,
        y,
        feature_indices=selected_indices,
    )

    print('\nStart RF classification with the 4 forward-selected features')
    rf_model_selected, rf_acc_selected, rf_best_params, rf_cv_score, rf_importances, rf_ranking = RF_classification(
        X,
        y,
        feature_indices=selected_indices,
    )

    print('\nGenerate learning curve for final SVM model')
    svm_curve = compute_learning_curve(
        estimator=svm_model_selected,
        X=X,
        y=y,
        feature_indices=selected_indices,
        train_ratios=TRAIN_RATIOS,
        n_repeats=N_REPEATS,
        random_state=42,
    )
    print_learning_curve_results(svm_curve, 'SVM')

    plot_learning_curve(
        svm_curve,
        model_name='SVM',
        metric='error',
        save_path='learning_curve_svm_error.png',
        show=False,
    )

    print('\nGenerate learning curve for final RF model')
    rf_curve = compute_learning_curve(
        estimator=rf_model_selected,
        X=X,
        y=y,
        feature_indices=selected_indices,
        train_ratios=TRAIN_RATIOS,
        n_repeats=N_REPEATS,
        random_state=42,
    )
    print_learning_curve_results(rf_curve, 'RF')
    
    plot_learning_curve(
        rf_curve,
        model_name='RF',
        metric='error',
        save_path='learning_curve_rf_error.png',
        show=False,
    )

