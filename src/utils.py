# Data wrangling and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pre-processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Tree-based Feature Selection
from sklearn.ensemble import RandomForestClassifier

# Modeling
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# Evaluation metrics
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Pipelines for Hyper-parameter tuning
from imblearn.pipeline import Pipeline

# Loading bar to print progress
from tqdm import tqdm

def classification_density(data, rows_subplot, columns_subplot, class_label, text_size=1.0, size=(18, 10)):
    # Get numerical features' names.
    col_names = data.select_dtypes(include=np.number).columns.tolist()
    # Set text size, and size of the plot (m columns x n rows)
    sns.set(font_scale=text_size)
    n_row = rows_subplot
    n_col = columns_subplot
    list_of_variables = col_names
    # Create instances of the figure and its axes.
    f, axes = plt.subplots(n_row, n_col, figsize=size)
    k = 0 # This counter helps indexing the features.
    # Fills the chart by row and column.
    for i in range(n_row): # For each row:
        for j in range(n_col): # For each column:
            # Create a density plot of variable k.
            sns.kdeplot(data[data[class_label] == 'Yes'][list_of_variables[k]],
                        label='Yes',
                        ax=axes[i, j])
            sns.kdeplot(data[data[class_label] == 'No'][list_of_variables[k]],
                        label='No',
                        ax=axes[i, j])
            k = k + 1
    # Plot legend outside the chart.
    plt.legend(title=class_label, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

def classification_boxplot(data, rows_subplot, columns_subplot, class_label, text_size=1.0, size=(12,12)):
    # Get numerical features' names.
    col_names = data.select_dtypes(include=np.number).columns.tolist()
    # Set text size, and size of the plot (m columns x n rows)
    sns.set(font_scale=text_size)
    n_row = rows_subplot
    n_col = columns_subplot
    list_of_variables = col_names
    # Create instances of the figure and its axes.
    f, axes = plt.subplots(n_row, n_col, figsize=size)
    k = 0 # This counter helps indexing the features.
    # Fills the chart by row and column.
    for i in range(n_row): # For each row:
        for j in range(n_col): # For each column:
            # Create a boxplot of variable k.
            sns.boxplot(x=list_of_variables[k],
                        y=class_label,
                        data=data,
                        ax=axes[i, j],
                        palette="Set3")
            # Do not set a y_label so the chart looks cleaner.
            axes[i, j].set_ylabel('')
            k = k + 1
    plt.show()

def iqr_outlier_detection(data, iqr_k=1.5, impute=False):
    # Get numerical features' names.
    col_names = data.select_dtypes(include=np.number).columns.tolist()
    # Calculate descriptive statistics only for numerical features.
    IQR = data[col_names].quantile(0.75) - data[col_names].quantile(0.25)
    lower_whisker = (data[col_names].median() - (IQR * iqr_k)).rename('lower_whisker')
    upper_whisker = (data[col_names].median() + (IQR * iqr_k)).rename('upper_whisker')
    # Build a dataframe of the decision boundaries using the (IQR x k) rule.
    boundaries = pd.concat([lower_whisker, upper_whisker], axis=1)

    if impute: # When impute is set to True:
        for i in range(len(boundaries)): # For each feature:
            # Get the name of the feature.
            feature = boundaries.index[i]
            # Find the lower and upper whisker of this feature.
            lower = boundaries.loc[feature, 'lower_whisker']
            upper = boundaries.loc[feature, 'upper_whisker']
            # Compute the median value of the feature.
            median = data[feature].median()
            # If the feature value isn't between the upper and the lower whiskers,
            # then replace with the median value of the feature.
            data[feature] = data[feature].mask(
                ~((data[feature] < upper) & (data[feature] > lower)), median)
    return boundaries

def encode_categorical_features(data, method='onehot'):
    # Split between numerical and categorical features.
    cat = data.select_dtypes(exclude=np.number)
    num = data.select_dtypes(include=np.number)
    # Store categorical features' names.
    col_names = data.select_dtypes(exclude=np.number).columns.tolist()
    # Based on the encoder selected, it creates an instance of it and fit/transforms the data.
    # Finally, the numerical and encoded categorical features are put together in a new dataframe that is returned.
    if method == 'onehot':
        ohencoder = OneHotEncoder(sparse=False)
        encoded_features = ohencoder.fit_transform(cat)
        cat_features = pd.DataFrame(data=encoded_features, columns=ohencoder.get_feature_names())
        new_df = pd.concat([num, cat_features], axis=1, sort=False)
    elif method == 'ordinal':
        ordinalencoder = OrdinalEncoder()
        encoded_features = ordinalencoder.fit_transform(cat)
        cat_features = pd.DataFrame(data=encoded_features, columns=col_names)
        new_df = pd.concat([num, cat_features], axis=1, sort=False)
    else:
        raise Exception("You should introduce a valid encoder: 'onehot' or 'ordinal'")
    return new_df

def split_and_scale_data(data, method='minmax', testsize=0.20, taget_class='RainTomorrow'):
    # Split dataframe into features and target class.
    y = data[taget_class]
    X = data.drop([taget_class], axis=1) # Drop target column from the original dataframe.
    # Perform train/test split of the data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y,
                                                        test_size=testsize, shuffle=True)

    # Scale the data based on the method selected.
    # Fit and transform using train, transform using test.
    # For the test set it only calls transform to avoid data leakage.
    if method == 'minmax':
        scaler = MinMaxScaler()
        # Fit and transform.
        scaled_data_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        # Transform using the data distribution of the training features.
        scaled_data_X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    elif method == 'standard':
        scaler = StandardScaler()
        # Fit and transform.
        scaled_data_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
        # Transform using the data distribution of the training features.
        scaled_data_X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    else:
        raise Exception("You should introduce a valid scaler: 'minmax' or 'standard'")
    return scaled_data_X_train, scaled_data_X_test, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

def feature_importance(X, y, estimators=600):
    # Fit Random Forest model
    model = RandomForestClassifier(
        n_estimators=estimators, n_jobs=-1,
        class_weight='balanced').fit(X, y)
    # Extract importance
    importances = model.feature_importances_
    # Reorder importance in decreasing order
    imp_order = np.argsort(importances)[::-1]
    # Get feature names and importance
    features = [X.columns[i] for i in imp_order]
    importance = [importances[i] for i in imp_order]
    return features, importance

def tree_based_feature_analysis(X_train, X_test, y_train, y_test):
    # Create empty lists to store the results at each iteration.
    acc = []
    f1 = []
    recall = []
    precision = []
    n_features = []
    features = []
    # Make a copy of the data so it can drop features while searching the best ones.
    # Training data:
    Xt = X_train.copy()
    yt = y_train.copy()
    # Testing data:
    Xts = X_test.copy()
    yts = y_test.copy()

    for i in tqdm(range(len(Xt.columns))):
        # Fit and predict on the training and testing data.
        clf = RandomForestClassifier(
        random_state=0, n_estimators=600,
        n_jobs=-1, class_weight='balanced')
        clf.fit(Xt, yt)
        y_pred = clf.predict(Xts)

        # Store evaluation metrics.
        acc.append(accuracy_score(y_pred=y_pred, y_true=yts))
        f1.append(f1_score(y_pred=y_pred, y_true=yts))
        recall.append(recall_score(y_pred=y_pred, y_true=yts))
        precision.append(precision_score(y_pred=y_pred, y_true=yts))
        n_features.append(len(Xt.columns))
        features.append(Xt.columns)

        # Get feature importances.
        importances = clf.feature_importances_
        # Get index of the feature with the least importance.
        drop = Xt.columns[np.argsort(importances)[0]]
        # Drop the feature from the training and test sets.
        Xt.drop(drop, axis=1, inplace=True)
        Xts.drop(drop, axis=1, inplace=True)
    return acc, f1, recall, precision, n_features, features

def stratified_cv(X, y, models, method, apply=None, k=10, weight0=None, weight1=None):
    # Create an instance of scikit-learn's StratifiedKFold.
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    # Create empty lists to store aggregated CV results.
    algorithm_name = []
    accuracy_s = []
    f1_s = []
    recall_s = []
    precision_s = []
    if (method not in ['sampling', 'regularization']): # Make sure a valid method is selected.
        raise Exception('Select a valid method. Options: regularization or sampling')
    for name, model in tqdm(models):
        # We initialize each model and CV with a clean set of lists to store the results at each iteration.
        accuracy = []
        f1 = []
        recall = []
        precision = []
        it = 1 # Count iterations
        for train_index, test_index in kf.split(X.values, y.values):
            if method == 'regularization':
                try: # Try running the model using all cores and a random seed (e.g. RandomForestClassifier)
                    clf = model(class_weight={0:weight0, 1:weight1}, random_state=0, n_jobs=-1)
                except: # If not possible:
                    try: # Try running the model without a random seed but using all cores (e.g. KNeighborsClassifier)
                        clf = model(class_weight={0: weight0, 1: weight1}, n_jobs=-1)
                    except: # If not possible:
                        try: # Try running the model sequentially with a random seed (e.g. DecisionTree)
                            clf = model(class_weight={0: weight0, 1: weight1}, random_state=0)
                        except: # If not possible:
                            # Try running the model without a random seed and sequentially (e.g. GaussianNB)
                            clf = model(class_weight={0: weight0, 1: weight1})
                clf.fit(X.values[train_index], y.values[train_index]) # Fit on training data.
            else:
                # Perform sampling on the training data for this split of cross fold.
                try: # Try using all cores.
                    sm = apply(random_state=0, n_jobs=-1)
                except: # If not possible run it sequentially.
                    sm = apply(random_state=0)
                X_t, y_t = sm.fit_resample(X.values[train_index], y.values[train_index])
                # Train the model on training data (same try/except sequence as above, but in this case it doesn't
                # perform regularization):
                try:
                    clf = model(random_state=0, n_jobs=-1)
                except:
                    try:
                        clf = model(n_jobs=-1)
                    except:
                        try:
                            clf = model(random_state=0)
                        except:
                            clf = model()
                clf.fit(X_t, y_t) # Fit on transformed data.

            # Predict on test data and store score metrics.
            results = clf.predict(X.values[test_index])
            accuracy.append(accuracy_score(y_pred=results, y_true=y.values[test_index]))
            f1.append(f1_score(y_pred=results, y_true=y.values[test_index]))
            precision.append(precision_score(y_pred=results, y_true=y.values[test_index]))
            recall.append(recall_score(y_pred=results, y_true=y.values[test_index]))
            it += 1
        # Calculate average metrics across all folds and store the results.
        algorithm_name.append(name)
        accuracy_s.append(np.mean(accuracy))
        f1_s.append(np.mean(f1))
        precision_s.append(np.mean(precision))
        recall_s.append(np.mean(recall))
    # Create a dataframe containing all the metrics (summary report).
    cv_summary = pd.DataFrame(data=({'algorithm_name': algorithm_name,
                                     'accuracy': accuracy_s,
                                     'f1': f1_s,
                                     'precision': precision_s,
                                     'recall': recall_s}))
    # Include method used to deal with imbalanced data.
    cv_summary['method'] = method
    # Include the number of folds in CV.
    cv_summary['folds'] = k
    if method == 'sampling':
        # If the method was sampling, include the name of the sampling technique
        cv_summary['apply'] = apply.__name__
    elif method == 'regularization':
        # If the method is regularization, include the weights of the cost matrix.
        cv_summary['weight0'] = weight0
        cv_summary['weight1'] = weight1
    return cv_summary

def hyperparameter_tuning(X, y, model, hyperpar_grid, sampling_method, folds=5, optimize='f1', refit_model=True):
    # Try creating an instance of the sampling technique that will use all cores during the optimization process.
    try:
        sampling = sampling_method(random_state=0, n_jobs=-1)
    except:
        sampling = sampling_method(random_state=0)
    # Create an imbalanced learn pipeline with the sampling technique and the model.
    pipeline = Pipeline([('sampling', sampling), ('model', model)])
    # Initiate the search.
    clf = GridSearchCV(pipeline, hyperpar_grid, cv=folds, verbose=0,
                       scoring=optimize, n_jobs=-1, refit=refit_model)
    clf.fit(X, y)
    # Return the best found parameters and the associated highest score.
    print('Parameters:', clf.best_params_, 'CV Score:', clf.best_score_)
    return clf.best_estimator_
