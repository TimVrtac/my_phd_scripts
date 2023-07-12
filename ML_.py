def test_all(Y):
    # PCA
    p = no_PC_opt
    no_FRFs = Y.shape[1] * Y.shape[2]
    H_ = np.reshape(Y[fr_min:fr_max], newshape=(fr_max - fr_min, no_FRFs), order='C')
    X_ = pca_.get_PCA_scores(H_).reshape(H_.shape[0] * p)

    # SVM - lin. jedro
    svm_lin_ = clf_lin.predict(np.log(abs(X_)[np.newaxis]))
    svm_pred_lin_ = translate_dummy(svm_lin_, dict_)

    # SVM - rad. jedro
    svm_rbf_ = clf_rbf.predict(np.log(abs(X_)[np.newaxis]))
    svm_pred_rbf_ = translate_dummy(svm_rbf_, dict_)

    # LDA
    lda_ = lda.predict(np.log(abs(X_)[np.newaxis]))
    lda_pred_ = translate_dummy(lda_, dict_)

    # QDA
    qda_ = qda.predict(np.log(abs(X_)[np.newaxis]))
    qda_pred_ = translate_dummy(qda_, dict_)

    # decision trees
    d_tree_ = d_tree.predict(np.log(abs(X_)[np.newaxis]))
    d_tree_pred_ = translate_dummy(d_tree_, dict_)

    # Ensamble - bagging
    bagging_ = bagging_cls.predict(np.log(abs(X_)[np.newaxis]))
    bagging_pred_ = translate_dummy(bagging_, dict_)

    # Ensamble - random forest
    random_f_ = random_f.predict(np.log(abs(X_)[np.newaxis]))
    random_f_pred_ = translate_dummy(random_f_, dict_)

    # Ensamble - Boosting
    adaboost_ = adaboost_clf.predict(np.log(abs(X_)[np.newaxis]))
    adaboodst_pred_ = translate_dummy(adaboost_, dict_)

    # k-Nearest Neighbors
    knn_ = knn.predict(np.log(abs(X_)[np.newaxis]))
    knn_pred_ = translate_dummy(knn_, dict_)

    # Nevronska mreža
    # Normalizacija podatkov
    X__norm = scaler.transform(abs(X_)[np.newaxis])
    # Napoved
    NN_ = np.argmax(model.predict(X__norm), axis=1)
    NN_pred_ = translate_dummy(NN_, dict_)

    results = {'svm (rad)': svm_pred_rbf_,
               'svm (lin)': svm_pred_lin_,
               'lda': lda_pred_,
               'qda': qda_pred_,
               'dec. tree': d_tree_pred_,
               'bagging': bagging_pred_,
               'rand. forest': random_f_pred_,
               'boosting': adaboodst_pred_,
               'KNN': knn_pred_,
               'NN': NN_pred_}

    return results
