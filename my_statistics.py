import matplotlib.pyplot as plt
import numpy as np
import prettytable
from tqdm.notebook import tqdm
import random


def MSE(y_pred, y_real):
    """
    Funcition calculates Mean Squared Error (MSE) for inserted data sets.
    :param: y_pred - predicted/aproximated values data set (np.array)
    :param: y_real - true values data set (np.array)
    :return: MSE
    """
    return (sum((y_pred-y_real)**2))/len(y_pred)


def mean(X):
    """
    Function calculates mean value for the data set X.
    Args:
        X: data set (numpy array)

    Returns: mean of X

    """
    return sum(X)/len(X)


def RSS(y_pred, y_real):
    """Function calculates and returns the Residual Sum of Squares (RSS).

    Args:
        y_pred (np.array): numeric array of predicted values
        y_real (np.array): numeric array of actual values

    Returns: RSS for input data
    """
    return sum((y_real-y_pred)**2)


def classification_error_rate(pred_, ref_):
    """Functions calculates classification error rate.

    Args:
        pred_ (numpy array): numpy array of predictions
        ref_ (numpy array): numpy array of reference output values

    Returns:
        float: classification error rate
    """
    n = ref_.shape[0]
    return 1/n*(n-np.count_nonzero(pred_ == ref_))


def stdev(X, type_='sample'):
    """
    Function calculates standard deviation for the data set X.
    Args:
        X: data set (numpy array)
        type_: 'sample' (N-1 in denominator) or 'popular' (N in denominator) (string)

    Returns: strandar deviation

    """
    den = None
    if type_ == 'popular':
        den = len(X)
    elif type_ == 'sample':
        den = len(X)-1
    return np.sqrt(sum((X-mean(X))**2)/den)


def cov(X, Y):
    """
    Function calculates the covariance for data sets X and Y.
    Args:
        X: data set (numpy array)
        Y: data set (numpy array)

    Returns: correlation

    """
    return sum((X-mean(X))*(Y-mean(Y)))/(len(X))


def corr(X, Y):
    """
    Function calculates the correlation for data sets X and Y.
    Args:
        X: data set (numpy array)
        Y: data set (numpy array)

    Returns: correlation

    """
    return cov(X, Y)/(stdev(X)*stdev(Y))


def euclidean(P1, P2):
    """
    Function calculates the Euclidean distance between points P1 and P2.
    Args:
        P1: point 1 coordinates (numpy array)
        P2: point 2 coordinates (numpy array)

    Returns: euclidean distance

    """
    return np.sqrt(sum(P1**2+P2**2))


def confusion_matrix(predictions, references, classes=(0, 1), print_=False):
    """
    Function calculates values of confusion matrix values together with sensitivity, specificity and overall accuracy
    and prints the result in the table form.
    Args:
        predictions: numeric array of predictions
        references: numeric array of referenca values
        classes: tuple of classes. Defoults to (0, 1)
        print_: if True, confusion matrix is printed

    Returns:

    """
    """
    
    :param predictions: array of predictions - values 0 or 1 (numpy array)
    :param references: array of references - values 0 or 1 (numpy array)

    return table, sensitivity, specificity, overall accuracy
    """
    TN = np.count_nonzero((predictions == classes[0]) & (references == classes[0]))
    TP = np.count_nonzero((predictions == classes[1]) & (references == classes[1]))
    FN = np.count_nonzero((predictions == classes[0]) & (references == classes[1]))
    FP = np.count_nonzero((predictions == classes[1]) & (references == classes[0]))
    sensitivity = TP/(TP + FN)
    specificity = TN/(TN + FP)
    overall = (TP + TN)/(TP+TN+FP+FN)
    table = prettytable.PrettyTable()
    table.field_names = ['True status', 'No', 'Yes', 'Sums']
    table.add_rows([['Prediciton', '', '', ''],
                    ['No', f' TN = {TN}', f' FN = {FN}', TN + FN],
                    ['Yes', f' FP = {FP}', f' TP = {TP}', TP + FP],
                    ['', f'specificity = {specificity*100:.2f} %', f'sensitivity = {sensitivity*100:.2f} %', ''],
                    ['Sums', TN + FN, FN + TP, TN+FN+TP+FP]])
    if print_:
        print(table)

    return table, sensitivity, specificity, overall


# STATISTICAL LEARNING
# ---------------------------------------------------------------------------------------------------------------------
# K-nearest neighbors (KNN)
def KNN(test_point_, input_data_, output_data, K, normalize=False, show=False, dummy=False):
    """
    # TODO: currently only for classification -> add regression
    Function applies K Nearest Neighbors method.
    Args:
        test_point_: test point location (np.array)
        input_data_: 1D or 2D array of feature values (1 column for individual feature)
        output_data: output for given input_data (np.array)
        K: number of points taken into account by KNN method
        normalize: iput_data is normalized to the values from 0 to 1
        show: Plot graph of points
        dummy: if True the dummy variables are generated for output_data

    Returns: category of test_point

    """
    if normalize:
        norm_basis = abs(input_data_).max(axis=0)  # values for normalization
        input_data = input_data_.copy()
        test_point = test_point_.copy()
        to_norm = norm_basis != 0  # predictors to be normalized
        input_data[:, to_norm] = input_data_[:, to_norm] / norm_basis[to_norm]
        test_point[to_norm] = test_point_[to_norm] / norm_basis[to_norm]                    
    else:
        input_data = input_data_.copy()
        test_point = test_point_.copy()

    # calculate distances
    distances = np.sqrt((np.sum(abs(input_data**2-test_point**2), axis=1)).astype(float))

    distances_sorted = sorted(distances)

    output_data_ = output_data.copy()
    if dummy:
        # dummy variables for output data:
        output_set = list(set(output_data))
        for i, j in enumerate(output_data):
            for k, l in enumerate(output_set):  # k predstavlja indeks elementa v naboru možnih izhodnih podatkov
                if output_data[i] == l:  # preverjanje če je posamezni element enak elementu l nabora možnih izhodnih
                    # podatkov
                    output_data_[i] = k  # pripis dummy variable k elementa l i-temu izhodnemu podatku

    # identify K closest points
    closest = output_data_[np.where(distances <= distances_sorted[K - 1])[0]]
    # determine class with highest probability
    count = np.unique(closest, return_counts=True)

    if show:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(test_point[0], test_point[1], test_point[2], facecolor='black', edgecolor='black', label='test')
        for i in range(len(output_data)):
            if i in np.where(distances <= distances_sorted[K - 1])[0]:
                ax.scatter(input_data[i, 0], input_data[i, 1], input_data[i, 2], facecolor='blue',
                           edgecolors=output_data[i], label='K closest')
            else:
                ax.scatter(input_data[i, 0], input_data[i, 1], input_data[i, 2], facecolor=output_data[i],
                           edgecolors=output_data[i], label='other points')
        ax.legend()

    return count[0][np.argmax(count[1])]


def k_fold_split(input_, output_, k):
    """Function splits input and output data for the purpose of k-fold Cross Validation.

    Args:
        input_ (numpy array): numeric array of input values
        output_ (numpy array): numeric array of output values
        k (int): number of folds

    Returns:
        list of numpy arrays: list of input folds
        list of numpy arrays: list of output folds
    """
    indices = list(np.arange(input_.shape[0], dtype=int))
    fold_size = int(input_.shape[0]/k)
    input_folds = []
    output_folds = []
    for i in range(k-1):
        k_ind = np.zeros(fold_size, dtype=int)
        for j in range(fold_size):
            k_ind[j] = random.choice(indices)
            indices.remove(k_ind[j])
        input_folds.append(np.array(input_[k_ind]))
        output_folds.append(np.array(output_[k_ind]))
    input_folds.append(input_[indices])
    output_folds.append(output_[indices])
    return input_folds, output_folds


def k_fold_CV_for_KNN_classification(inputs_, outputs_, k=10, K=np.arange(1, 10), find_K_opt=False):
    """Function performs k-fold Cross Validation for the purpose of optimal K value determination in KNN machine
     learning method.

    Args:
        inputs_ (numpy array): numeric array of input values (n×p matrix) where n is sample size and p is number of
                               predictors.
        outputs_ (numpy array): numeric array of output values
        k (int, optional): number of folds. Defaults to 10.
        K (list/numpy array, optional): Possible K values. Defaults to np.arange(1,10).
        find_K_opt (bool, optional): if True the optimal K value is returned. Defaults to False.

    Returns:
        list: error rates corresponding to the input K values
        int (optional): optimal K value
    """
    # data split
    input_f, output_f = k_fold_split(inputs_, outputs_, k=k)
    error_rates = []
    for k_ in tqdm(K, desc=f'Main loop'):
        errors_ = []
        for i in tqdm(range(k), desc=f'loop for K = {k_}: ', leave=False):
            k_pred = np.zeros(input_f[i].shape[0])
            pred_fold_ = input_f[i]
            ref_fold_ = output_f[i]
            ind_list = list(np.arange(k))
            ind_list.remove(i)
            input_folds_ = input_f[ind_list]
            input_folds_ = np.concatenate(input_folds_.squeeze(), axis=0)
            output_folds_ = output_f[ind_list]
            output_folds_ = np.concatenate(output_folds_.squeeze(), axis=0)
            for j in range(input_f[i].shape[0]):
                k_pred[j] = KNN(pred_fold_[j], input_folds_, output_folds_, K=k_, normalize=True)
            errors_.append(classification_error_rate(k_pred, ref_fold_))
        error_rates.append(np.average(errors_))
    if find_K_opt:
        K_opt = K[np.argmin(np.array(error_rates))]
        return error_rates, K_opt
    return error_rates


# ----------------------------------------------------------------------------------------------------------------------------------------------
# Linear Discriminant Analysis (LDA)
def lin_discriminant_function(x, μ, Σ, π):
    """Function calculates the value of linear disciriminant function for given x.

    Args:
        x (np.array): vector of predictor values for which we would like to perform a prediction
        μ (np.array): vector of mean values for individual predictors
        Σ (np.array): covariance matrix
        π (float): prior probability

    Returns:
        float: values of discriminant function
    """
    return x.T@np.linalg.inv(Σ)@μ-0.5*μ.T@np.linalg.inv(Σ)@μ + np.log(π)


def linear_discriminant_analysis(input_data, predictors, outputs):
    """Function applies LDA method. It learns from predictors and outputs data and make predictions for input_data.

    Args:
        input_data (np.array): predictor values for which we would like to perform predictions (1 column for 1
                               predictor) (1D or 2D np.array)
        predictors (np.array): numeric array of predictor values (1 column for 1 predictor) (1D or 2D np.array)
        outputs (np.array): numeric array of known system outputs
        
    Returns:
        [type]: [description]
    """
    categories = np.unique(outputs)
    mu = np.zeros((len(categories), predictors.shape[1]))
    pi = np.zeros(len(categories))
    for j, i in enumerate(categories):
        avg = np.average(predictors[outputs == i], axis=0)
        mu[j, :] = avg
        pi[j] = predictors[outputs == i].shape[0]/predictors.shape[0]
    p_matrix = np.cov(predictors.astype(float).T)
    # discriminant function
    predictions_ = []
    for j, i in enumerate(input_data):
        delta_ = []
        for k in range(len(categories)):
            delta_.append(lin_discriminant_function(i, mu[k, :], p_matrix, pi[k]))
        category_ind = np.argmax(delta_)
        predictions_.append(categories[category_ind])
    
    return np.array(predictions_)


# LDA method for 2 class preditions which enables Bayes decision boundary modification
def LDA_decision_boundary_mod(input_data, predictors, outputs,
                              decision_boundary=(1, .5)):
    """LDE for 2 class prediction which enables Bayes decision boundary modification.
    
    Args:
        input_data (np.array): predictor values for which we would like to perfrom predictions (1 column for 1
                               predictor)
                               (1D or 2D np.array)
        predictors (np.array): numeric array of predictor values (1 column for 1 predictor) (1D or 2D np.array)
        outputs (np.array): numeric array of known system outputs
        decision_boundary (tuple of int,float; optional): Bayes decision boundary modification - tuple
                                (class for which we are setting decision 
                                boundary [0 or 1], decision boundary
                                [0 to 1])
    Returns:
        numpy array: predicition by LDA
    :param 
    """
    categories = np.unique(outputs)
    mu = np.zeros((len(categories), predictors.shape[1]))
    pi = np.zeros(len(categories))
    for j, i in enumerate(categories):
        avg = np.average(predictors[outputs == i], axis=0)
        mu[j, :] = avg
        pi[j] = predictors[outputs == i].shape[0]/predictors.shape[0]
    p_matrix = np.cov(predictors.astype(float).T)
    # discriminant function
    predictions_ = []
    for j, i in enumerate(input_data):
        delta_ = []
        for k in range(len(categories)):
            delta_.append(lin_discriminant_function(i, mu[k, :],
                                                    p_matrix, pi[k]))
        delta_ratio = delta_[decision_boundary[0]]/(delta_[decision_boundary[0]] + delta_[not decision_boundary[0]])
        if delta_ratio > decision_boundary[1]:
            predictions_.append(categories[decision_boundary[0]])
        else:
            predictions_.append(categories[int(not decision_boundary[0])])
    return np.array(predictions_)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# Quadratic Discriminant Analysis (QDA)
def quad_discriminant_function(x, μ, Σ, π):
    """Function calculates the value of quadratic disciriminant function for given x.

    Args:
        x (np.array): vector of predictor values for which we would like to perform a prediction
        μ (np.array): vector of mean values for individual predictors
        Σ (np.array): covariance matrix
        π (float): prior probability

    Returns:
        float: values of discriminant function
    """
    return -0.5*x.T@np.linalg.inv(Σ)@x + x.T@np.linalg.inv(Σ)@μ-0.5*μ.T@np.linalg.inv(Σ)@μ - \
        0.5*np.log(np.linalg.det(Σ)) + np.log(π)


def quadratic_discriminant_analysis(input_data, predictors, outputs):
    """Function applies QDA method for making predictions

    Args:
        input_data (np.array): predictor values for which we would like to perfrom predictions (1 column for 1
        predictor) (1D or 2D np.array)
        predictors (np.array): numeric array of predictor values (1 column for 1 predictor) (1D or 2D np.array)
        outputs (np.array): numeric array of known system outputs
        
    Returns:
        numpy array: predicition by QDA
    """
    categories = np.unique(outputs)
    mu = np.zeros((len(categories), predictors.shape[1]))
    pi = np.zeros(len(categories))
    p_matrix = np.zeros((len(categories), predictors.shape[1], predictors.shape[1]))
    for j, i in enumerate(categories):
        avg = np.average(predictors[outputs == i], axis=0)
        mu[j, :] = avg
        pi[j] = predictors[outputs == i].shape[0]/predictors.shape[0]
        p_matrix[j, :, :] = np.cov(predictors[outputs == i].astype(float).T)
    # discriminant function
    predictions_ = []
    for j, i in enumerate(tqdm(input_data)):
        delta_ = []
        for k in range(len(categories)):
            delta_.append(quad_discriminant_function(i, mu[k, :], p_matrix[k, :, :], pi[k]))
        category_ind = np.argmax(delta_)
        predictions_.append(categories[category_ind])
    
    return np.array(predictions_)


# general k-Fold CV
def k_fold_CV(inputs_, outputs_, k, K, prediction_model):
    # TODO: passing additional arguments into prediction_model function
    """Function performs k-fold Cross Validation for the purpose of optimal K value determination of prediction_model function

    Args:
        inputs_ (numpy array): numeric array of input values (n×p matrix) where n is sample size and p is number of
                               predictors.
        outputs_ (numpy array): numeric array of output values
        k (int): number of folds. Defaults to 10.
        K (list/numpy array): Parameter for which we search the value which provides minimum error.
                                        Possible K values.
        prediction_model (function): function with parameters: x_test, x_train, y_train which performs predictions for given
                                     input data.

    Returns:
        list: error rates corresponding to the input K values
    """
    # data split
    input_f, output_f = k_fold_split(inputs_, outputs_, k=k)
    error_rates = []
    for k_ in tqdm(K, desc=f'Main loop'):
        errors_ = []
        for i in tqdm(range(k), desc=f'loop for K = {k_}: ', leave=False):
            k_pred = np.zeros(input_f[i].shape[0])
            pred_fold_ = input_f[i]
            ref_fold_ = output_f[i]
            ind_list = list(np.arange(k))
            ind_list.remove(i)
            input_folds_ = input_f[ind_list]
            input_folds_ = np.concatenate(input_folds_.squeeze(), axis=0)
            output_folds_ = output_f[ind_list]
            output_folds_ = np.concatenate(output_folds_.squeeze(), axis=0)
            for j in range(input_f[i].shape[0]):
                k_pred[j] = prediction_model(pred_fold_[j], input_folds_, output_folds_, K=k_,)
            errors_.append(classification_error_rate(k_pred, ref_fold_))
        error_rates.append(np.average(errors_))
    return error_rates

# TODO: add logistic reg.
# TODO: implement shrinking methods to remove redundant predictors: best subset selection, stepwise selection,
# ridge regression, the lasso, dimension reduction
# TODO: add GAMs along with different shapes of functions f(X) - polynomial regression, step function, regression
# splines, smoothing splines, local regression
