import matplotlib.pyplot as plt
import numpy as np
import prettytable
from tqdm.notebook import tqdm
import random
from IPython.display import clear_output
from tools import H
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold


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


# PCA
# ---------------------------------------------------------------------------------------------------------------------
# PCA
class PCA:
    def __init__(self, H_matrix, p=None):
        self.H = H_matrix
        self.p = p
        self.col_avg, self.col_std, self.H_adj, self.eigvals, self.eigvecs, self.Cmatch = self.get_reduction_matrix()

    def get_reduction_matrix(self):
        col_avg = self.H.mean(axis=0)  # average column value
        print('Average over columns - done')
        col_std = np.sum((self.H - col_avg)**2, axis=0)/self.H.shape[0]  # standard deviation for over columns
        print('Standard deviation over columns - done')
        H_adj = (self.H - col_avg)/np.sqrt(col_std*self.H.shape[0])  # H matrix adjustment
        print('Adjusted H matrix - done')
        C = np.conj(H_adj.T) @ H_adj  # Correlation matrix
        print('Correlation matrix - done')
        eigvals, eigvecs = np.linalg.eig(C)  # eignevalue problem solution
        print('Eigenproblem - done')
        clear_output()
        return col_avg, col_std, H_adj, eigvals, eigvecs, C

    def get_projection_matrix(self, H_=None, p=None):
        if p is not None:
            self.p = p
        if H_ is None:
            H_adj_ = self.H_adj
        else:
            H_adj_ = (H_ - self.col_avg) / np.sqrt(self.col_std * self.H.shape[0])
        return np.einsum('ki,ij->kj', H_adj_, self.eigvecs[:, :self.p])

    def reconstruct(self, H_=None):
        if H_ is None:
            H_ = self.H
        H_adj_R = np.einsum('ij,kj->ik', self.get_projection_matrix(H_), self.eigvecs[:, :self.p])
        return H_adj_R * (np.sqrt(self.col_std*self.H.shape[0])) + self.col_avg

    def plot_results(self, H_, plot_PC_scores, plot_reconstruction, no_scores, i):
        """TODO: bolj posplošiti - trenutno samo za izhodiščni H"""
        if plot_reconstruction and plot_PC_scores:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax_1, ax_2 = ax[0], ax[1]
        elif plot_reconstruction:
            fig, ax_2 = plt.subplots(1, 1)
        elif plot_PC_scores:
            fig, ax_1 = plt.subplots(1, 1)

        if plot_PC_scores:
            ax_1.semilogy(abs(self.get_PCA_scores(H_)[:, :no_scores]))
            ax_1.set_title('PCA scores')

        if plot_reconstruction:
            ax_2.semilogy(abs(self.reconstruct())[:, i], label='reconstruction', c='k', lw=3)
            ax_2.semilogy(abs(self.H)[:, i], '--', label='original', c='y');
            ax_2.legend()
            ax_2.grid()


def PCA_old(H_matrix, p, reconstruct=False, compare=False, i=None, show_scores=False, no_scores=None):
    """
    Principal component analysis implementation. H must be of shape: n×m, where n (rows) is a
    number of frequency points and m (columns) is a number of channels.
    H: FRF matrix
    p: numer of principal components kept after reduction
    reconstruct: reconstruction of FRF after dim. reduction
    compare: plot graph with comparison of original vs reconstructed FRF
    i: index of plotted FRF in comparison graph
    show_scores: plot pca results (scores)
    no_scores: number of scores plotted (must be less than p)
    """

    row_avg = H_matrix.mean(axis=1)  # average row value
    H_adj = (H_matrix - row_avg[:, np.newaxis])  # H matrix adjustment
    eigvals, eigvecs = np.linalg.eig(np.conjugate(H_adj.T) @ H_adj)  # eignevalue problem solution

    PCA_scores = H_adj @ eigvecs[:, :p]  # PCA

    if compare and show_scores:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax_1, ax_2 = ax[0], ax[1]
    elif compare:
        fig, ax_2 = plt.subplots(1, 1)
    elif show_scores:
        fig, ax_1 = plt.subplots(1, 1)
    if show_scores:
        ax_1.semilogy(abs(PCA_scores[:, :no_scores]))
        ax_1.set_title('PCA scores')

    if reconstruct:  # reconstruction
        PCA_rec = PCA_scores @ np.conjugate(eigvecs[:, :p].T) + row_avg[:, np.newaxis]

        if compare:
            ax_2.semilogy(abs(PCA_rec)[:, i], label='reconstruction', c='k', lw=3)
            ax_2.semilogy(abs(H_matrix)[:, i], '--', label='original', c='y');
            ax_2.legend()
            ax_2.grid()
        return PCA_scores, PCA_rec
    return PCA_scores


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
    """Function performs k-fold Cross Validation for the purpose of optimal K value determination of prediction_model
     function.

    Args:
        inputs_ (numpy array): numeric array of input values (n×p matrix) where n is sample size and p is number of
                               predictors.
        outputs_ (numpy array): numeric array of output values
        k (int): number of folds. Defaults to 10.
        K (list/numpy array): Parameter for which we search the value which provides minimum error.
                                        Possible K values.
        prediction_model (function): function with parameters: x_test, x_train, y_train which performs predictions for
                                     given input data.

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


# ----------------------------------------------------------------------------------------------------------------------------------------------
# Ordinal classification
class OrdinalClassifier():
    """
    Application of ordinal classification algorithm accroding from: E. Frank, M. Hall - "A simple approach to ordinal classification".
    The algorithm extends the binary classification algorithm to the ordinal classification by transforming the labels.

    Parameters:
    clf: sklearn classifier object (scikit binary classifier that supports predict_proba method, i.e. it returns probabilities)
    class_list: list of class labels (ordinal classification labels)
    """
    def __init__ (self, clf, class_list: list):
        self.clf = clf
        self.class_list = class_list
            

    def fit(self, X: np.array, y:np.array, **clf_params):
        """Function fits the classifier to the provided training data.

        Args:
            X (2D np.array): Training data
            y (1D np.array): Training labels
            clf_params (dict): Parameters specific for the provided classifier.

        Returns:
            None
        """
        y_ord = self.ordinal_encoder(y)
        self.cls_list = [self.clf(**clf_params) for i in range(y_ord.shape[1])]
        self.cls_fitted = [self.cls_list[i].fit(X, y_ord[:,i]) for i in range(y_ord.shape[1])]
        

    def predict(self, X:np.array, return_prob:bool=False):
        """Function predicts the labels for the provided data.

        Args:
            X (np.array): Data for which the labels are to be predicted.
            return_prob (bool, optional): If true, probabilities for each class of each sample are returned. Defaults to False.

        Returns:
            predictions (np.array): Predicted labels for the provided data.
            probabilites (np.array): Probabilities for each class of each sample.
        """
        true_ind = [np.argwhere(clf.classes_==1)[0] for clf in self.cls_fitted]
        probability_array = np.array([clf.predict_proba(X)[:,true_ind[i_]].squeeze() for i_, clf in enumerate(self.cls_fitted)]).T
        model_probabilities = self.get_class_probabilities(probability_array)
        class_ind_array = np.argmax(model_probabilities, axis=1)
        pred_ = [self.class_list[ind_] for ind_ in class_ind_array]
        if return_prob:
            return np.array(pred_), model_probabilities
        else:
            return np.array(pred_)

    def get_class_probabilities(self, probability_array):
        """Based on the individual ML model output probabilities, the function calculates the probabilities for each class.

        Args:
            probability_array (np.array): Array of probabilities returned by the individual ML model.

        Returns:
            np.array: Array of probabilities for each class.
        """
        prob_ = np.zeros((probability_array.shape[0], probability_array.shape[1]+1))
        prob_[:,0] = np.ones(probability_array.shape[0]) - probability_array[:,0]
        prob_[:,-1] = probability_array[:,-1]
        for i in range(1, len(self.class_list)-1):
            prob_[:,i] = probability_array[:,i-1] - probability_array[:,i]
        return prob_

    def ordinal_encoder(self, y):
        """ 
        Transformation of the labels to the ordinal classification format.

        Args:
            y (np.array): Labels to be transformed

        Returns:
            np.array: Transformed labels
        """
        new_class_lims = self.class_list[:-1]
        y_ord = np.zeros((y.shape[0], len(new_class_lims)))
        for i in range(len(new_class_lims)):
            y_ord[:,i] = y > new_class_lims[i]
        return y_ord
    
    def cross_validate(self, X:np.array, y:np.array, y_acc:np.array, valid_2: tuple=None , n_splits: int=10, n_repeats: int=10, random_state: int=42, leave_bar=True, bar_desc=None, **clf_params):
        """Function performs cross-validation to test the accuracy of the classifier.

        Args:
            Main dataset:
                X (np.array): Data used for the cross-validation
                y (np.array): Labels used for the cross-validation
                y_acc (np.array): Accurate values of the labels

            Additional validation data:
                valid_2 (tuple, optional): Additional dataset for validation of the ML algorithm; this data is not used in the training process. Form: (X_2, y_2, y_acc_2) Defaults to None.

            Cross-validation parameters:
                n_splits (int, optional): Number of folds in k-fold CV. Defaults to 10.
                n_repeats (int, optional): Number of repeated CVs. Defaults to 10.
                random_state (int, optional): Random state for selection of samples in individual folds. Defaults to 42.
        """
        cv_split = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        self.n_splits = cv_split.get_n_splits(X, y)
        self.cv_results = {'Accurate values': [], 'Labels': [], 'Predictions': []}
        if valid_2 is not None:
            self.cv_results_2 = {'Accurate values': [], 'Labels': [], 'Predictions': []}
        for train, test in tqdm(cv_split.split(X, y), total=self.n_splits, leave=leave_bar, desc=bar_desc):
            X_train_, y_train_ = X[train], y[train]
            X_test_, y_test_, y_acc_ = X[test], y[test], y_acc[test]
            self.fit(X_train_, y_train_, **clf_params)
            #self.cv_results.append({'Accurate value': y_acc_, 'Label': y_test_, 'Prediction': self.predict(X_test_)})
            self.cv_results['Accurate values'].extend(y_acc_)
            self.cv_results['Labels'].extend(y_test_)
            self.cv_results['Predictions'].extend(self.predict(X_test_))
            if valid_2 is not None:
                self.cv_results_2['Accurate values'].extend(valid_2[2])
                self.cv_results_2['Labels'].extend(valid_2[1])
                self.cv_results_2['Predictions'].extend(self.predict(valid_2[0]))
                #self.cv_results_2.append({'Accurate value': valid_2[2], 'Label': valid_2[1], 'Prediction': self.predict(valid_2[0])})
            self.cls_list = [clone(_) for _ in self.cls_list]
        self.cv_results['Test score'] = self.get_test_score(self.cv_results['Labels'], self.cv_results['Predictions'])
        self.cv_results['Total error'] = self.get_total_error(self.cv_results['Labels'], self.cv_results['Predictions'])
        self.cv_results['Squared error'] = self.get_mean_squared_error(self.cv_results['Labels'], self.cv_results['Predictions'])
        if valid_2 is not None:
            self.cv_results_2['Test score'] = self.get_test_score(self.cv_results_2['Labels'], self.cv_results_2['Predictions'])
            self.cv_results_2['Total error'] = self.get_total_error(self.cv_results_2['Labels'], self.cv_results_2['Predictions'])
            self.cv_results_2['Squared error'] = self.get_mean_squared_error(self.cv_results_2['Labels'], self.cv_results_2['Predictions'])

    @staticmethod
    def get_test_score(y, y_pred):
        """Function calculates the accuracy of the classifier.

        Args:
            y (np.array): True labels
            y_pred (np.array): Predicted labels

        Returns:
            float: Accuracy of the classifier
        """
        return np.mean(np.array(y) == np.array(y_pred))
    
    @staticmethod
    def get_total_error(y, y_pred):
        """Function calculates the total error of the classifier.

        Returns:
            float: Total error of the classifier
        """
        return sum(abs(np.array(y) - np.array(y_pred))) / len(y)
    
    @staticmethod
    def get_mean_squared_error(y, y_pred):
        """Function calculates the squared error of the classifier.

        Returns:
            float: Squared error of the classifier
        """
        return sum((np.array(y) - np.array(y_pred))**2) / len(y)

# TODO: add logistic reg.
# TODO: implement shrinking methods to remove redundant predictors: best subset selection, stepwise selection,
# ridge regression, the lasso, dimension reduction
# TODO: add GAMs along with different shapes of functions f(X) - polynomial regression, step function, regression
# splines, smoothing splines, local regression
