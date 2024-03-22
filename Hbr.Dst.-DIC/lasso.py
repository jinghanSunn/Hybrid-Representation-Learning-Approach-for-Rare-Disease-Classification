import numpy as np
from sklearn.linear_model import LassoCV

def lasso_v1(X, y):
    lasso = LassoCV(cv=5, random_state=0, n_alphas=1000)
    lasso.fit(X, y)

    selected_features = np.where(lasso.coef_ != 0)[0]
    X_selected = X[:, selected_features]

    return selected_features

features_moco = np.load('./tsne_fig/feature_moco.npy')
features_srl = np.load('./tsne_fig/feature_srl.npy')
features_hbr = np.load('./tsne_fig/feature_hbrdic.npy')
feature_sup = np.save('./tsne_fig/feature_sup.npy')




