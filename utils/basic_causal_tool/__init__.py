import numpy as np


def get_skeleton_from_stable_pc(X, return_time=False):
    """ if return_time is True, return estimate_skeleton, cg.PC_elapsed
    """
    import networkx as nx
    from sklearn.utils import check_array
    from causallearn.search.ConstraintBased.PC import pc

    X = check_array(X)
    cg = pc(data=X, show_progress=False)
    cg.to_nx_skeleton()
    estimate_skeleton = nx.to_numpy_array(cg.nx_skel)

    if return_time is True:
        return estimate_skeleton, cg.PC_elapsed
    else:
        return estimate_skeleton


def residual_by_linreg(X, y):
    """ Usually perform pairwise or multiple regression in practice.
        Shape of residuals: (n, 1)
    """
    from sklearn.linear_model import LinearRegression
    from utils.data_processing import check_vector

    X = check_vector(X)
    y = check_vector(y)

    model = LinearRegression()
    model.fit(X, y)
    y_hat = check_vector(model.predict(X))
    residual = y - y_hat

    return residual


def fisher_hsic(X, R):
    """
    - xj ind.of r(j -> neighbor(j))
    where only r(j -> neighbor(j)) is a vector, and wants to test whether xj is an exogenous variable;

    - xj(neighbor) ind.of r(neighbor(j) -> j)
    where only xj(neighbor) is a vector, and wants to test whether xj is a sink variable.

    :param X: numpy array, (n,1) or (n,d)
    :param R: numpy array, (n,1) or (n,d)
    :return: hsic p value(when a pair test) or fisher p value
    """
    from scipy.stats import chi2
    from lingam.hsic import hsic_test_gamma

    fisher_stat = 0
    d_x = X.shape[1]
    d_r = R.shape[1]

    if d_x == 1 and d_r == 1:  # pair indendent test
        _, hsic_p_value = hsic_test_gamma(X, R)
        return hsic_p_value  # It has nothing to do with statistics, chi-square distribution

    elif d_x == 1 and d_r > 1:  # find exogenous variable
        for i in range(d_r):
            _, hsic_p_value = hsic_test_gamma(X, R[:, i])

            # punish if p value just 0
            fisher_stat += np.inf if hsic_p_value == 0 else -2 * np.log(hsic_p_value)

            if fisher_stat > np.inf:
                break

        fisher_p_value = 1 - chi2.cdf(fisher_stat, df=2 * d_r)
        return fisher_p_value

    elif d_x > 1 and d_r == 1:  # find sink variable
        for i in range(d_x):
            _, hsic_p_value = hsic_test_gamma(X[:, i], R)

            # punish if p value just 0
            fisher_stat += np.inf if hsic_p_value == 0 else -2 * np.log(hsic_p_value)

            if fisher_stat > np.inf:
                break

        fisher_p_value = 1 - chi2.cdf(fisher_stat, df=2 * d_r)
        return fisher_p_value

    else:
        raise ValueError("Data shape (columns) should be large or eaqual to 1.")


