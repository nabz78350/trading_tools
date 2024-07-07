
import pandas as pd
import numpy as np
from scipy.stats import norm
from quantstats.stats import sharpe
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch, set_module
import matplotlib.pyplot as plt
from tqdm import tqdm
import linearmodels as lm

import statsmodels.api as sm


import scipy.stats as st
def get_conf_interval(X):
    # Critical t values
    alpha = 0.05
    tc = st.t.ppf(1 - alpha / 2, df=len(X) - 6)
    print('t-critico:', tc)
    print('')
    return tc

def get_conf_interval_params(results,X):
    tc = get_conf_interval(X)
    
    # Calculating the confidence intervals with 95% two-tailed confidence
    intervalo0 = results.params - results.bse * tc
    intervalo1 = results.params + results.bse * tc
    intervalo = pd.concat([intervalo0, intervalo1], axis=1)
    intervalo.columns = ['Lower', 'Upper']
    return intervalo


from statsmodels.sandbox.regression.predstd import wls_prediction_std


def fit_ols_gls(X, Y, add_constant=True):
    # Step 1: Add constant if required
    if add_constant:
        X = sm.add_constant(X)

    # Step 2: Fit OLS model
    ols_model = sm.OLS(Y, X)
    ols_results = ols_model.fit()
    
    # Step 3: Compute residuals
    residuals = ols_results.resid
    
    # Step 4: Estimate heteroskedasticity
    # We assume the covariance matrix of the residuals is diagonal for simplicity
    # More complex structures can be estimated using different methods
    sigma_hat = np.var(residuals)  # An estimate of sigma^2
    Sigma = np.diag(sigma_hat * np.ones(len(residuals)))  # Covariance matrix of residuals
    
    # Step 5: Compute the transformation matrix P and its inverse
    P = cholesky(Sigma, lower=True)
    P_inv = np.linalg.inv(P)
    
    # Step 6: Transform the model
    Y_star = P_inv @ Y
    X_star = P_inv @ X

    gls_model = sm.GLS(Y_star, X_star,sigma = Sigma)
    gls_results = gls_model.fit()
    
    return ols_results, gls_results

def recursive_least_squares(X, Y, add_constant=True, standardize=True, linear_constraint=None):
    # Standardize X and Y if required
    if standardize:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        Y_mean = np.mean(Y)
        Y_std = np.std(Y)
        
        X = (X - X_mean) / X_std
        Y = (Y - Y_mean) / Y_std

    # Add constant if required
    if add_constant:
        X = sm.add_constant(X)
    
    
    
    # Fit the model with or without the constraint
    if linear_constraint:
        model = sm.RecursiveLS(Y, X,constraints= linear_constraint)
        result = model.fit()
    else:
        model = sm.RecursiveLS(Y, X)
        result = model.fit()
    
    
    return result 

def plot_pred_ci(results,X_test,Y_test):
    # Generating the confidence intervals
    prstd, iv_l, iv_u = wls_prediction_std(results, exog=X_test)

    # Estimating forecasted values
    beta = results.params
    Y_hat = np.dot(X_test, beta)

    # Generating the x-axis of the graphs
    x = X_test.index

    # Graphing the forecast
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(x, Y_test, 'o', label="true data")
    ax.plot(x, Y_hat, 'b-', label="Estimate")
    ax.plot(x, iv_u, 'r--')
    ax.plot(x, iv_l, 'r--')
    ax.legend(loc='best')

def compute_pred(results,X_test):

    Y_hat = results.predict(X_test)
    return pd.DataFrame({"pred":Y_hat})

from statsmodels.tools.eval_measures import mse, rmse, meanabs
from sklearn.metrics import r2_score

def make_results_table(results, X_test, Y_test):
    """
    Creates a table with columns for true values and predicted values based on statsmodels results.
    Parameters:
    - results: The results object from a statsmodels regression.
    - X_test: The exogenous variable matrix for testing.
    - Y_test: The actual test values.

    Returns:
    - A pandas DataFrame with columns 'True' and 'Pred'.
    """
    Y_pred = results.predict(X_test)
    assert len(Y_pred) == len(Y_test)
    results_table = pd.DataFrame({
        'Pred': Y_pred
    })
    results_table['True'] = Y_test.iloc[:,0]

    return results_table
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def compute_metrics(results_table):
    """
    Computes various accuracy and metrics for a table with 'True' and 'Pred' columns.

    Parameters:
    - results_table: A pandas DataFrame with columns 'True' and 'Pred'.

    Returns:
    - A pandas DataFrame with the computed metrics (MSE, RMSE, MAE, R2, U).
    """
    Y_true = results_table['True']
    Y_pred = results_table['Pred']
    
    mse_val = mean_squared_error(Y_true, Y_pred)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(Y_true, Y_pred)
    r2_val = r2_score(Y_true, Y_pred)
    u_val = rmse_val / np.sqrt(mean_squared_error(Y_true, np.zeros_like(Y_pred)))

    metrics = {
        'MSE': [mse_val],
        'RMSE': [rmse_val],
        'MAE': [mae_val],
        'R2': [r2_val],
        'U': [u_val]
    }

    metrics_df = pd.DataFrame(metrics)
    return metrics_df


import statsmodels.stats.diagnostic as sm_diagnostic

def perform_ramsey_test(results, power=3, use_f=True, test_type='fitted'):
    """
    Performs the Ramsey RESET test for model specification.

    Parameters:
    - results: The results object from a statsmodels regression.
    - power: The power to which to raise the fitted values (default is 3).
    - use_f: Whether to use the F-statistic (default is True).
    - test_type: The type of test to perform (default is 'fitted').

    Returns:
    - A tuple containing the test statistic and the p-value.
    """
    reset_test = sm_diagnostic.linear_reset(results, power=power, use_f=use_f, test_type=test_type)
    return reset_test


from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

def create_polynomial_variables(X, degree=3, input_features=None, cross_multiply=True):
    """
    Calculates the coefficients with polynomial variables.

    Parameters:
    - X: The input DataFrame containing the features.
    - degree: The degree of the polynomial features (default is 3).
    - input_features: List of feature names to be used for naming the polynomial features (default is None).
    - cross_multiply: Boolean indicating whether to cross-multiply features or only add polynomial terms of each feature independently (default is True).

    Returns:
    - A DataFrame with the polynomial features.
    """
    const_col = None
    if 'const' in X.columns:
        print('intercept detected')
        const_col = X['const']
        X = X.drop(columns=['const'])

    if cross_multiply:
        poly = PolynomialFeatures(degree=degree, interaction_only=False,include_bias=False)
        X_pol = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(input_features=input_features)
        X_pol_df = pd.DataFrame(X_pol, index=X.index, columns=feature_names)
    else:
        X_pol_dfs = []
        for col in X.columns:
            print(col)
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_col = X[[col]]
            X_pol_col = poly.fit_transform(X_col)
            feature_names = poly.get_feature_names_out(input_features=[col])
            X_pol_dfs.append(pd.DataFrame(X_pol_col, index=X.index, columns=feature_names))
        
        X_pol_df = pd.concat(X_pol_dfs, axis=1)

    if const_col is not None:
        X_pol_df['const'] = const_col

    return X_pol_df


# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic as sm_diagnostic

def perform_ramsey_test(model, labels=['F-stats', 'p-value'], alpha=0.05):
    """
    Performs the Ramsey RESET test for the fitted model to check if more polynomial terms should be added.

    Parameters:
    - model: The fitted linear model (statsmodels).
    - labels: List of labels for the test statistics (default is ['F-stats', 'p-value']).
    - alpha: Significance level for the test (default is 0.05).

    Returns:
    - A DataFrame with the test statistics and a recommendation on whether to add more polynomial terms.
    """
    reset_result = sm_diagnostic.linear_reset(model, power=2, use_f=True, test_type='fitted')
    stats = np.round([reset_result.statistic[0, 0], reset_result.pvalue], 4)
    stats_df = pd.DataFrame(stats, index=labels, columns=['Value'])

    if stats[1] < alpha:
        recommendation = "Reject the null hypothesis. Consider adding more polynomial terms."
    else:
        recommendation = "Fail to reject the null hypothesis. No need to add more polynomial terms."

    return stats_df, recommendation



# %% [markdown]
# Ramsey test

# %%
import statsmodels.api as sm
import warnings
import statsmodels.stats.diagnostic as sm_diagnostic


def ramsey_test(results,power:int = 3,test_type:str = "exog",use_f:bool = False):
    reset = sm_diagnostic.linear_reset(results, power=power,test_type=test_type,use_f=use_f)
    labels = ['F-stats:', 'p-value:']
    print(reset.statistic)
    
    stats = ([reset.statistic, reset.pvalue])
    return(pd.DataFrame(stats, index=labels, columns=['Value']))


# %%
import statsmodels.api as sm
import warnings
import statsmodels.stats.diagnostic as sm_diagnostic
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

# %%
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

def perform_granger_causality_test(data, causing_column, caused_column, max_lag):
    """
    Perform Granger causality test.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.
    causing_column (str): The column name of the variable that is hypothesized to cause.
    caused_column (str): The column name of the variable that is hypothesized to be caused.
    max_lag (int): The maximum number of lags to be tested.

    Returns:
    None: Prints the results of the Granger causality tests.
    """
    exog = data[[causing_column, caused_column]].diff().dropna()

    print(f'Test {causing_column} Granger-causes {caused_column}')
    res = grangercausalitytests(exog, maxlag=[max_lag], verbose=True)
    return res



# %%
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

def perform_granger_causality_test(data, causing_column, caused_column, max_lag):
    """
    Perform Granger causality test.

    Parameters:
    data (pd.DataFrame): The dataframe containing the data.
    causing_column (str): The column name of the variable that is hypothesized to cause.
    caused_column (str): The column name of the variable that is hypothesized to be caused.
    max_lag (int): The maximum number of lags to be tested.

    Returns:
    None: Prints the results of the Granger causality tests and whether the causation is significant.
    """
    def is_significant(results):
        # Check if p-values of F-test are significant at 0.05 level
        for key in results.keys():
            test_result = results[key][0]['ssr_ftest']
            if test_result[1] < 0.05:
                return True
        return False

    exog = data[[causing_column, caused_column]].diff().dropna()

    print(f'Test {causing_column} Granger-causes {caused_column}')
    results = grangercausalitytests(exog, maxlag=[max_lag], verbose=True)
    significant = is_significant(results)
    print(f'Significant: {"Yes" if significant else "No"}\n')


# %%
############################################################
# Defining the Forward Regression function
############################################################
def forward_regression(X, y, criterion="pvalue", threshold=0.05, verbose=False):
    """
    Select the variables that estimate the best model using stepwise forward regression.

    Parameters:
    -----------
    X : DataFrame of shape (n_samples, n_features)
        Features matrix, where n_samples is the number of samples and n_features is the number of features.
    y : Series of shape (n_samples, 1)
        Target vector, where n_samples is the number of samples.
    criterion : str, can be {'pvalue', 'AIC', 'SIC', 'R2' or 'R2_A'}
        The default is 'pvalue'. The criterion used to select the best features:
        
        - 'pvalue': select the features based on p-values.
        - 'AIC': select the features based on Lowest Akaike Information Criterion.
        - 'SIC': select the features based on Lowest Schwarz Information Criterion.
        - 'R2': select the features based on Highest R Squared.
        - 'R2_A': select the features based on Highest Adjusted R Squared.
    threshold : scalar, optional
        Is the maximum p-value for each variable that will be accepted in the model. The default is 0.05.
    verbose : bool, optional
        Enable verbose output. The default is False.

    Returns:
    --------
    value : list
        A list of the variables that produce the best model.

    Raises:
    -------
    ValueError
        When the value cannot be calculated.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a DataFrame")

    if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series):
        raise ValueError("y must be a column DataFrame")

    if isinstance(y, pd.DataFrame):
        if y.shape[0] > 1 and y.shape[1] > 1:
            raise ValueError("y must be a column DataFrame")

    included = []
    aic = 1e10
    sic = 1e10
    r2 = -1e10
    r2_a = -1e10

    if criterion == "pvalue":
        value = 0
        while value <= threshold:
            excluded = list(set(X.columns) - set(included))
            best_pvalue = 999999
            new_feature = None
            for i in excluded:
                factors = included + [i]
                X1 = X[factors]
                X1 = sm.add_constant(X1)
                results = sm.OLS(y, X1).fit()
                cond_1 = results.pvalues.index != "const"
                new_pvalues = results.pvalues[cond_1]
                new_pvalue = new_pvalues.max()
                if best_pvalue > new_pvalue and cond_1.sum() != 0 and new_pvalue < threshold:
                    best_pvalue = new_pvalue
                    best_value = results.pvalues[cond_1].max()
                    new_feature = i
                    pvalues = new_pvalues.copy()

            value = pvalues[pvalues.index != "const"].max()
            if new_feature is None:
                break

            included.append(new_feature)

            if verbose:
                print(f"Add {new_feature} with p-value {best_pvalue:.6f}")

    else:
        excluded = X.columns.tolist()
        for i in range(X.shape[1]):
            j=0
            value = None
            for i in excluded:
                factors = included.copy()
                factors.append(i)
                X1 = X[factors]
                X1 = sm.add_constant(X1)
                results = sm.OLS(y, X1).fit()
                if criterion == "AIC":
                    if results.aic < aic:
                        value = i
                        aic = results.aic
                if criterion == "SIC":
                    if results.bic < sic:
                        value = i
                        sic = results.bic
                if criterion == "R2":
                    if results.rsquared > r2:
                        value = i
                        r2 = results.rsquared
                if criterion == "R2_A":
                    if results.rsquared_adj > r2_a:
                        value = i
                        r2_a = results.rsquared_adj

                j += 1
                if j == len(excluded):
                    if value is None:
                        break
                    else:
                        excluded.remove(value)
                        included.append(value)
                        if verbose:
                            if criterion == "AIC":
                                print(f"Add {value} with AIC {results.aic:.6f}")
                            elif criterion == "SIC":
                                print(f"Add {value} with SIC {results.bic:.6f}")
                            elif criterion == "R2":
                                print(f"Add {value} with R2 {results.rsquared:.6f}")
                            elif criterion == "R2_A":
                                print(f"Add {value} with Adjusted R2 {results.rsquared_adj:.6f}")

    return included




# %%
def backward_regression(X, y, criterion="pvalue", threshold=0.05, verbose=False):
    """
    Select the variables that estimate the best model using stepwise backward regression.

    Parameters:
    -----------
    X : DataFrame of shape (n_samples, n_features)
        Features matrix, where n_samples is the number of samples and n_features is the number of features.
    y : Series of shape (n_samples, 1)
        Target vector, where n_samples is the number of samples.
    criterion : str, can be {'pvalue', 'AIC', 'SIC', 'R2' or 'R2_A'}
        The default is 'pvalue'. The criterion used to select the best features:
        
        - 'pvalue': select the features based on p-values.
        - 'AIC': select the features based on Lowest Akaike Information Criterion.
        - 'SIC': select the features based on Lowest Schwarz Information Criterion.
        - 'R2': select the features based on Highest R Squared.
        - 'R2_A': select the features based on Highest Adjusted R Squared.
    threshold : scalar, optional
        Is the maximum p-value for each variable that will be accepted in the model. The default is 0.05.
    verbose : bool, optional
        Enable verbose output. The default is False.

    Returns:
    --------
    value : list
        A list of the variables that produce the best model.

    Raises:
    -------
    ValueError
        When the value cannot be calculated.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a DataFrame")

    if not isinstance(y, pd.DataFrame) and not isinstance(y, pd.Series):
        raise ValueError("y must be a column DataFrame")

    if isinstance(y, pd.DataFrame):
        if y.shape[0] > 1 and y.shape[1] > 1:
            raise ValueError("y must be a column DataFrame")

    X1 = sm.add_constant(X)
    results = sm.OLS(y, X1).fit()
    pvalues = results.pvalues
    aic = results.aic
    sic = results.bic
    r2 = results.rsquared
    r2_a = results.rsquared_adj

    included = pvalues.index.tolist()
    excluded = ["const"]

    if criterion == "pvalue":
        while pvalues[pvalues.index != "const"].max() > threshold:
            factors = pvalues[pvalues.index.isin(excluded) == False].index.tolist()
            print(factors)
            X1 = X[factors]
            X1 = sm.add_constant(X1)
            results = sm.OLS(y, X1).fit()
            pvalues = results.pvalues
            pvalues = pvalues[~pvalues.index.isin(["const"])]
            excluded.append(pvalues.idxmax())
            if verbose and pvalues.max() > threshold:
                print(f"Drop {pvalues.idxmax()} with p-value {pvalues.max():.6f}")
                included.remove(pvalues.idxmax())
        included.remove('const')
    else:
        included = pvalues.index.tolist()
        included.remove("const")
        for j in range(X.shape[1]):
            j = 0
            value = None
            for i in included:
                factors = included.copy()
                factors.remove(i)
                X1 = X[factors]
                X1 = sm.add_constant(X1)
                results = sm.OLS(y, X1).fit()
                if criterion == "AIC":
                    if results.aic < aic:
                        value = i
                        aic = results.aic
                elif criterion == "SIC":
                    if results.bic < sic:
                        value = i
                        sic = results.bic
                elif criterion == "R2":
                    if results.rsquared > r2:
                        value = i
                        r2 = results.rsquared
                elif criterion == "R2_A":
                    if results.rsquared_adj > r2_a:
                        value = i
                        r2_a = results.rsquared_adj

                j += 1
                if j == len(included):
                    if value is None:
                        break
                    else:
                        included.remove(value)
                        if verbose:
                            if criterion == "AIC":
                                print(f"Drop {value} with AIC {results.aic:.6f}")
                            elif criterion == "SIC":
                                print(f"Drop {value} with SIC {results.bic:.6f}")
                            elif criterion == "R2":
                                print(f"Drop {value} with R2 {results.rsquared:.6f}")
                            elif criterion == "R2_A":
                                print(f"Drop {value} with Adjusted R2 {results.rsquared_adj:.6f}")

    return included




# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in the dataset.

    Parameters:
    X (array-like): Feature data.
    exclude_constant (bool): Whether to exclude the constant column (default is True).

    Returns:
    list: VIF values for each feature.
    """
    vif = []
    for i in range(X.shape[1]):
        vif.append(variance_inflation_factor(X, i))
    out = pd.DataFrame({"VIF":vif},index = X.columns.tolist())
    
    out ['No multicollinearity'] = np.sign(out['VIF']-5) <0
    out ['Multicollinearity'] = np.sign(out['VIF']-10) >0
    
    return out



import numpy as np

def calculate_condition_number(X):
    """
    Calculate the Condition Number (CN) for the given feature data.
    
    The CN is used to diagnose multicollinearity in regression analysis.
    It is defined as the square root of the ratio of the largest eigenvalue 
    to the smallest eigenvalue of the X'X matrix. Higher CN values indicate 
    stronger multicollinearity:
    - 10 ≤ CN ≤ 30 suggests moderate multicollinearity.
    - CN > 30 suggests high multicollinearity.
    
    Parameters:
    X (array-like): Feature data.
    Returns:
    float: Condition Number (CN).
    """

    lambdas, _ = np.linalg.eig(X.T @ X)
    lambda_max = np.max(lambdas)
    lambda_min = np.min(lambdas)

    CN = np.sqrt(lambda_max / lambda_min)
    return CN

# %%
import numpy as np
import scipy.stats as st

def farrar_glauber_orthogonality_test(X, alpha=0.05):
    """
    Perform the Farrar-Glauber Orthogonality Test for multicollinearity.

    This test evaluates the orthogonality of the regressors based on the correlation matrix.
    The hypotheses are:
    - H0: The X are orthogonal to each other.
    - H1: The X are not orthogonal to each other.

    Parameters:
    X (array-like): Feature data.
    alpha (float): Significance level for the test (default is 0.05).

    Returns:
    dict: A dictionary with chi2 value, p-value, chi2 table value, and interpretation.
    """
    n = X.shape[0]
    k = X.shape[1]

    # Degree of freedom
    df = k * (k - 1) / 2

    # Correlation matrix of exogenous variables
    R = np.corrcoef(X, rowvar=False)

    # Calculating the chi2 statistic
    chi2 = -(n - 1 - (2 * k + 5) / 6) * np.log(np.linalg.det(R))
    p_value = 1 - st.chi2.cdf(chi2, df=df)
    chi2_table = st.chi2.ppf(1 - alpha, df=df)

    # Interpretation
    if chi2 >= chi2_table:
        interpretation = "Reject H0: The regressors are not orthogonal to each other."
    else:
        interpretation = "Fail to reject H0: The regressors are orthogonal to each other."

    return {
        'Chi2 Calculated': chi2,
        'P-value': p_value,
        'Chi2 Table': chi2_table,
        'Interpretation': interpretation
    }

def detect_max_lag_acf(residuals, alpha=0.5):
    # Step 1: Compute ACF of residuals with confidence intervals
    acf_values, confint = acf(residuals, alpha=alpha, fft=False, nlags=len(residuals) - 1)
    
    max_lag = 0
    for lag in range(1, len(acf_values)):
        if confint[lag, 0] < acf_values[lag] < confint[lag, 1]:
            max_lag = lag
            break
    
    return max_lag



def fit_glsar(X, Y, add_constant=True, alpha=0.05):
    # Add constant if required
    if add_constant:
        X = sm.add_constant(X)
    
    # Step 1: Fit initial OLS model
    ols_model = sm.OLS(Y, X)
    ols_results = ols_model.fit()
    
    # Step 2: Compute residuals from OLS fit
    residuals = ols_results.resid
    rho = detect_max_lag_acf(residuals) 
    if rho > 0:
        print('GLSAR')
        glsar_model = sm.GLSAR(Y, X, rho=rho)
        glsar_results = glsar_model.fit()
    else:
        glsar_model = sm.GLS(Y, X)
        glsar_results = glsar_model.fit()
    
    return ols_results, glsar_results, rho




# %%
import numpy as np
import scipy.stats as st
from statsmodels.stats.outliers_influence import variance_inflation_factor

def f_test(X2, alpha=0.05):
    """
    Performs the F Test to determine which variable is more closely aligned
    with the other variables, similar to calculating the Variance Inflation Factor (VIF).
    
    Args:
    X2 (numpy.ndarray): 2D array with the independent variables.
    alpha (float): Significance level for hypothesis testing. Default is 0.05.
    
    Returns:
    dict: Contains the calculated F statistic, p-value, F critical value,
          and the position of the variable with the maximum R^2.
          
    if F-calc < F-table the null hypothesis is accepted, the presence of multicollinearity in the model is rejected
    """
    # Calculating the R2s for the auxiliary regressions
    R2s = []
    for i in range(X2.shape[1]):
        vif = variance_inflation_factor(X2, i)
        R2 = 1 - 1 / vif
        R2s.append(R2)

    R2_max = np.max(R2s)
    n = X2.shape[0]
    k = X2.shape[1]

    # Degrees of freedom
    dfn = k - 1
    dfd = n - k

    # Calculating the F statistic
    F = (R2_max / (k - 1)) / ((1 - R2_max) / (n - k))
    p_value = 1 - st.f.cdf(F, dfn, dfd)
    F_critical = st.f.ppf(1 - alpha, dfn, dfd)

    # Finding the variable with maximum R2
    pos = np.argmax(R2s)

    results = {
        'F_statistic': F,
        'p_value': p_value,
        'F_critical': F_critical,
        'variable_position': pos
    }
    
    return results



# %%
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_regression(X, Y, n_components=2, standardize=True,add_constant = False):
    """
    Performs Principal Component Regression (PCR) on the given dataset.
    
    Args:
    X (numpy.ndarray): 2D array with the independent variables.
    Y (numpy.ndarray): 1D array with the dependent variable.
    n_components (int or float): Number of principal components to keep. 
                                 If an integer, keeps the first 'n_components' components.
                                 If a float between 0 and 1, keeps enough components to explain
                                 the 'n_components' fraction of the variance.
    standardize (bool): Whether to standardize the variables before performing PCA. Default is True.
    
    Returns:
    dict: Contains the PCA components, principal components of the data, 
          and the regression results summary.
    """
    # Standardizing the variables
    if standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
    else:
        X_std = X

    # Performing PCA
    pca = PCA(n_components=n_components)
    Z_p = pd.DataFrame(pca.fit_transform(X_std),columns = [f"PC{i+1}" for i in range(n_components)],index = Y.index)
    
    if add_constant == True:
        Z_p = sm.add_constant(Z_p)
    V_p = pca.components_.T

    # Performing regression with the principal components
    res = sm.OLS(Y, Z_p).fit()
    beta_pc = res.params[1:]

    results = {
        'PCA_components': V_p,
        'Principal_components': Z_p,
        'Regression_summary': res.summary(),
        'PCA_betas': beta_pc
    }
    
    return results


import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

def plsr_regression(X, Y, n_components=2, standardize=True, add_constant=False):
    """
    Performs Partial Least Squares Regression (PLSR) on the given dataset.

    Args:
    X (numpy.ndarray): 2D array with the independent variables.
    Y (numpy.ndarray): 1D array with the dependent variable.
    n_components (int): Number of latent variables to keep.
    standardize (bool): Whether to standardize the variables before performing PLS. Default is True.
    
    Returns:
    dict: Contains the PLS components, scores of the data, and the regression results summary.
    """
    # Standardizing the variables
    if standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
    else:
        X_std = X

    # Performing PLS
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_std, Y)
    Z_p = pd.DataFrame(pls.transform(X_std), columns=[f"LV{i+1}" for i in range(n_components)], index=pd.Index(Y.index))

    if add_constant:
        Z_p = sm.add_constant(Z_p)

    # Performing regression with the latent variables
    res = sm.OLS(Y, Z_p).fit()

    results = {
        'PLS_components': pls.x_weights_.T,  # Weight vectors
        'Scores': Z_p,  # Latent variables
        'Regression_summary': res.summary(),
        'PLS_coefficients': res.params
    }

    return results


import numpy as np
import pandas as pd
import statsmodels.api as sm

def ridge_regression(X, Y, lambda_min=0,lambda_max=10,add_constant = True,standardize= True):
    """
    Perform Ridge Regression for a range of delta values and return the coefficients
    transformed to original variables.

    Parameters:
    X (np.ndarray): The input features, where the first column is a constant (intercept).
    Y (np.ndarray): The target variable.
    deltas (np.ndarray, optional): An array of delta values for Ridge Regression regularization.
                                   If None, defaults to np.linspace(0, 10, 11).

    Returns:
    pd.DataFrame: DataFrame containing Ridge Regression coefficients for different delta values.
    pd.DataFrame: DataFrame containing transformed Ridge coefficients for original variables.
    """
    if lambda_min is None or lambda_max is None:
        deltas = np.linspace(0, 10, 11)
    else :
        deltas = np.linspace(lambda_min, lambda_max, 10).round(2)


    if standardize :
        Y = (Y - Y.mean())/ Y.std()
        X_std = X.copy()
        X_mean = np.mean(X_std, axis=0)
        X_std = (X_std - X_mean) / np.std(X_std, axis=0, ddof=1)
    else :
        Y = Y
        X_std = X.copy()
    if add_constant:
        X_std = np.column_stack((np.ones(X_std.shape[0]), X_std))  # Add intercept back
    else:
         X_std = X_std
         print(X_std)
         
    betas_ridge = {}

    # Calculating the beta of the Ridge Regression
    for delta in deltas:
        model = sm.OLS((Y - Y.mean())/Y.std(), X_std).fit_regularized(alpha=delta, L1_wt=0)
        betas_ridge[f'delta {delta}'] = model.params

    ridge_df = pd.DataFrame(betas_ridge).T
    ridge_df = pd.DataFrame(betas_ridge).T

    if add_constant :
        ridge_df.columns = ["const"] + X.columns.tolist()
    else :
        ridge_df.columns =  X.columns.tolist()
    return ridge_df


# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm


def lasso_regression(X, Y, lambda_min=0, lambda_max=0.1, add_constant=True, standardize=False):
    """
    Perform Lasso Regression for a range of delta values and return the coefficients
    transformed to original variables.

    Parameters:
    X (np.ndarray): The input features, where the first column is a constant (intercept).
    Y (np.ndarray): The target variable.
    lambda_min (float): The minimum value of the regularization parameter lambda.
    lambda_max (float): The maximum value of the regularization parameter lambda.
    deltas (np.ndarray, optional): An array of delta values for Lasso Regression regularization.
                                   If None, defaults to np.linspace(lambda_min, lambda_max, 10).
    add_constant (bool): Whether to add a constant intercept term. Default is True.
    standardize (bool): Whether to standardize the variables before performing Lasso. Default is False.

    Returns:
    pd.DataFrame: DataFrame containing Lasso Regression coefficients for different delta values.
    """
    deltas = np.linspace(lambda_min, lambda_max, 10).round(2)
        
    # Standardizing the features (excluding the intercept term)
    if standardize:
        X_std = X.copy()
        X_mean = np.mean(X_std, axis=0)
        X_std = (X_std - X_mean) / np.std(X_std, axis=0, ddof=1)
    else:
        X_std = X.copy()
    if add_constant:
        X_std = np.column_stack((np.ones(X_std.shape[0]), X_std))  # Add intercept back
    else:
        X_std = X_std
    
    betas_lasso = {}

    # Calculating the beta of the Lasso Regression
    for delta in deltas:
        model = sm.OLS(Y - Y.mean(), X_std).fit_regularized(alpha=delta, L1_wt=1)
        betas_lasso[f'delta {delta}'] = model.params

    lasso_df = pd.DataFrame(betas_lasso).T

    if add_constant:
        lasso_df.columns = ["const"] + X.columns.tolist()
    else:
        lasso_df.columns = X.columns.tolist()

    return lasso_df



import numpy as np
import statsmodels.api as sm
def recursive_least_squares(X, Y, add_constant=True, standardize=True, linear_constraint=None):
    # Standardize X and Y if required
    if standardize:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        Y_mean = np.mean(Y)
        Y_std = np.std(Y)
        
        X = (X - X_mean) / X_std
        Y = (Y - Y_mean) / Y_std

    # Add constant if required
    if add_constant:
        X = sm.add_constant(X)
    
    
    
    # Fit the model with or without the constraint
    if linear_constraint:
        model = sm.RecursiveLS(Y, X,constraints= linear_constraint)
        result = model.fit()
    else:
        model = sm.RecursiveLS(Y, X)
        result = model.fit()
    
    
    return result


import numpy as np
import statsmodels.api as sm
import scipy.stats as st

def chow_test(X, Y, data_index,threshold):
    """
    Perform the Chow test for structural breaks in a linear regression model and record the break dates
    in the top 10% of Chow statistics.

    Parameters:
    X (np.ndarray): The input features matrix.
    Y (np.ndarray): The target variable vector.
    data_index (pd.Index): The index of the data, used to determine the break date.

    Returns:
    list: A list of dictionaries, each containing the break period, break date, F-statistic, and p-value.
    """
    n, k = X.shape
    chow_statistics = []

    for i in range(k + 1, n - (k + 1)):
        model = sm.OLS(Y, X).fit()
        SCR = model.ssr
        model_1 = sm.OLS(Y[:i], X[:i]).fit()
        SCR_1 = model_1.ssr
        model_2 = sm.OLS(Y[i:], X[i:]).fit()
        SCR_2 = model_2.ssr
        chow_i = ((SCR - (SCR_1 + SCR_2)) / k) / ((SCR_1 + SCR_2) / (n - 2 * k))
        chow_statistics.append((i, chow_i))
    
    # Sort the chow_statistics by Chow value and keep the top 10%
    chow_statistics.sort(key=lambda x: x[1], reverse=True)
    top_chow_statistics = chow_statistics[:threshold]

    results = []
    for i, chow_i in top_chow_statistics:
        p_value = 1 - st.f.cdf(chow_i, k, n - 2 * k)
        result = {
            'Break period': i,
            'Break date': data_index[i].date(),
            'F-stat': np.round(chow_i, 4),
            'p-value': np.round(p_value, 4)
        }
        results.append(result)

    return results





import numpy as np
import statsmodels.api as sm
from scipy.linalg import cholesky

def fit_ols_gls(X, Y, add_constant=True):
    # Step 1: Add constant if required
    if add_constant:
        X = sm.add_constant(X)

    # Step 2: Fit OLS model
    ols_model = sm.OLS(Y, X)
    ols_results = ols_model.fit()
    
    # Step 3: Compute residuals
    residuals = ols_results.resid
    
    # Step 4: Estimate heteroskedasticity
    # We assume the covariance matrix of the residuals is diagonal for simplicity
    # More complex structures can be estimated using different methods
    sigma_hat = np.var(residuals)  # An estimate of sigma^2
    Sigma = np.diag(sigma_hat * np.ones(len(residuals)))  # Covariance matrix of residuals
    
    # Step 5: Compute the transformation matrix P and its inverse
    P = cholesky(Sigma, lower=True)
    P_inv = np.linalg.inv(P)
    
    # Step 6: Transform the model
    Y_star = P_inv @ Y
    X_star = P_inv @ X

    gls_model = sm.GLS(Y_star, X_star,sigma = Sigma)
    gls_results = gls_model.fit()
    
    return ols_results, gls_results





import numpy as np
import statsmodels.api as sm
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf,acf
import matplotlib.pyplot as plt


def detect_max_lag_acf(residuals, alpha=0.5):
    # Step 1: Compute ACF of residuals with confidence intervals
    acf_values, confint = acf(residuals, alpha=alpha, fft=False, nlags=len(residuals) - 1)
    
    max_lag = 0
    for lag in range(1, len(acf_values)):
        if confint[lag, 0] < acf_values[lag] < confint[lag, 1]:
            max_lag = lag
            break
    
    return max_lag



def fit_glsar(X, Y, add_constant=True, alpha=0.05):
    # Add constant if required
    if add_constant:
        X = sm.add_constant(X)
    
    # Step 1: Fit initial OLS model
    ols_model = sm.OLS(Y, X)
    ols_results = ols_model.fit()
    
    # Step 2: Compute residuals from OLS fit
    residuals = ols_results.resid
    rho = detect_max_lag_acf(residuals) 
    if rho > 0:
        print('GLSAR')
        glsar_model = sm.GLSAR(Y, X, rho=rho)
        glsar_results = glsar_model.fit()
    else:
        glsar_model = sm.GLS(Y, X)
        glsar_results = glsar_model.fit()
    
    return ols_results, glsar_results, rho



# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

def fit_feasible_gls(X, Y, add_constant=True):
    X_arr = X.values
    Y_arr = Y.values
    # Add constant if required
    if add_constant:
        X = sm.add_constant(X)
    
    n = X.shape[0]

    # Step 1: Fit an OLS model
    ols_model = sm.OLS(Y, X).fit()
    ols_residuals = ols_model.resid.values

    # Step 2: Estimate the rho parameter based on OLS residuals
    rho_model = sm.OLS(ols_residuals[1:], ols_residuals[:-1]).fit()
    rho_hat = rho_model.params[0]

    # Step 3: Estimate variances based on OLS residuals
    if add_constant:
        var_model = sm.OLS(np.log(ols_residuals**2), X.values[:, 1:]).fit()
    else :
        var_model = sm.OLS(np.log(ols_residuals**2), X.values).fit()
    e_hat = np.sqrt(np.exp(var_model.fittedvalues))

    sigma_hat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sigma_hat[i, j] = (e_hat[i] * e_hat[j]) / (1 - rho_hat**2) * (rho_hat**abs(i - j))

    # Step 5: Fit GLS model with the estimated covariance matrix
    gls_model = sm.GLS(Y, X, sigma=sigma_hat).fit()

    print("OLS Model Summary:")
    print(ols_model.summary())
    print("\nGLS Model Summary:")
    print(gls_model.summary())

    return ols_model, gls_model


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math

def plot_residuals(results):
    # Extract residuals and fitted values
    residuals = results.resid
    fitted_values = results.fittedvalues
    exog = results.model.exog
    exog_names = results.model.exog_names

    # Determine the number of features
    num_features = exog.shape[1] - 1 if 'const' in exog_names else exog.shape[1]

    # Calculate the number of rows and columns for the subplots
    num_plots = num_features + 2
    num_cols = math.ceil(math.sqrt(num_plots))
    num_rows = math.ceil(num_plots / num_cols)
    
    # Set up the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows))
    axes = axes.flatten()  # Flatten to easily access each subplot

    # Plot the residuals in sequence
    axes[0].scatter(range(len(residuals)), residuals)
    axes[0].set_title("Run-Sequence Plot")
    axes[0].set_xlabel("Observation")
    axes[0].set_ylabel("Residuals")

    # Plot residuals vs fitted values
    axes[1].scatter(fitted_values, residuals)
    axes[1].set_title("Residuals vs Fitted")
    axes[1].set_xlabel("Fitted Values")
    axes[1].set_ylabel("Residuals")

    # Plot residuals vs each feature
    start_index = 2
    for i in range(num_features):
        feature_index = i + (1 if 'const' in exog_names else 0)
        axes[start_index + i].scatter(exog[:, feature_index], residuals)
        axes[start_index + i].set_title(f"Residuals vs {exog_names[feature_index]}")
        axes[start_index + i].set_xlabel(exog_names[feature_index])
        axes[start_index + i].set_ylabel("Residuals")

    # Hide any unused subplots
    for j in range(start_index + num_features, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout
    plt.tight_layout()
    plt.show()



def fit_wls(X, Y, add_constant=True,correct_heteroskedasticity: bool = True, custom_weights: pd.Series= None):
    
    
    if not correct_heteroskedasticity and  custom_weights is None:
        raise ValueError( "Specify a weighting method")

    if custom_weights is not None and correct_heteroskedasticity:
        raise ValueError(" Choose between custom or heteroskedasticity correction")
    if add_constant:
        X = sm.add_constant(X)
    
    n = X.shape[0]

    # Step 1: Fit an OLS model
    if correct_heteroskedasticity:
        ols_model = sm.OLS(Y, X).fit()
        ols_residuals = ols_model.resid
        log_resid = sm.OLS(np.log(ols_residuals**2), X).fit().fittedvalues
        h_est = np.exp(log_resid)
        wls = sm.WLS(Y, X, weights = 1.0/ h_est).fit()
        
    else:
        # Custom weights handling
        if custom_weights is not None:
            if len(custom_weights) != len(Y):
                raise ValueError("Custom weights must have the same length as Y")
            
            # Normalize the custom weights to ensure higher values mean more weight
            custom_weights = custom_weights / custom_weights.sum()
            print(custom_weights,custom_weights.sum())
            wls = sm.WLS(Y, X, weights=custom_weights.values).fit()
        else:
            raise ValueError("Custom weights provided but empty")
    return wls


import pandas as pd
import statsmodels.stats.diagnostic as sm_diagnostic

def goldfeld_quandt_test(y, x, idx=0, alternative="two-sided"):
    """
    Performs the Goldfeld-Quandt test for heteroscedasticity.

    Parameters:
    y (array-like): Dependent variable
    x (array-like): Independent variable(s)
    idx (int, optional): The index of the variable used to sort the data. Default is 0.
    alternative (str, optional): The alternative hypothesis. Options are "two-sided", "increasing", or "decreasing".
                                  Default is "two-sided".

    Returns:
    pd.DataFrame: A DataFrame containing the F statistic, p-value, and the type of test.

    Interpretation:
    The Goldfeld-Quandt test checks for heteroscedasticity by comparing the variances of two subsamples.
    - Null hypothesis (H0): The variances of the errors of both subsamples are equal.
    - Alternative hypothesis (H1): The variances of the errors of both subsamples are not equal.
    If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis, indicating heteroscedasticity.
    """

    GQ = sm_diagnostic.het_goldfeldquandt(y=y, x=x, alternative=alternative, idx=idx)
    labels = ['F statistic', 'p-value', 'type']
    result_df = pd.DataFrame(GQ, index=labels, columns=['Value'])
    
    return result_df


import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm

def spearman_test(residuals, X, significance_level=0.05):
    """
    Performs the Spearman test for heteroscedasticity.

    Parameters:
    residuals (array-like): Residuals of the model
    X (pd.DataFrame): Independent variable(s)
    significance_level (float, optional): Significance level for the hypothesis test. Default is 0.05.

    Returns:
    pd.DataFrame: A DataFrame containing the Spearman's rho statistic, p-value, and interpretation for each independent variable.

    Interpretation:
    The Spearman test checks for heteroscedasticity by calculating the Spearman correlation between 
    the absolute values of the model residuals and each independent variable.
    - Null hypothesis (H0): The errors are homoscedastic (no correlation between residuals and independent variables).
    - Alternative hypothesis (H1): The errors are heteroscedastic (correlation exists between residuals and independent variables).
    If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis, indicating heteroscedasticity.
    """

    n = X.shape[0]
    data = {'S_p': [], 'p-value': [], 'interpretation': []}

    for i in range(X.shape[1]):
        R_ie, _ = st.spearmanr(np.abs(residuals), X.iloc[:, i])
        S_p = R_ie * np.sqrt((n - 2) / (1 - R_ie ** 2))
        p_value = 1 - st.t.cdf(S_p, df=n - 2)
        interpretation = 'Heteroscedastic' if p_value < significance_level else 'Homoscedastic'
        
        data['S_p'].append(S_p)
        data['p-value'].append(p_value)
        data['interpretation'].append(interpretation)

    result_df = pd.DataFrame(data, index=X.columns)
    
    return result_df






import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic as sm_diagnostic

def white_test(residuals, exog):
    """
    Performs the White test for heteroscedasticity.

    Parameters:
    residuals (array-like): Residuals of the model
    exog (pd.DataFrame): Independent variable(s)

    Returns:
    pd.DataFrame: A DataFrame containing the LM statistic, p-value, and interpretation.

    Interpretation:
    The White test checks for heteroscedasticity by examining the relationship between the squared residuals 
    and the independent variables, their polynomial terms, and their interactions.
    - Null hypothesis (H0): The coefficients of the independent variables are equal to zero, indicating homoscedasticity.
    - Alternative hypothesis (H1): At least one coefficient is not equal to zero, indicating heteroscedasticity.
    If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis, indicating heteroscedasticity.
    """
    print(exog.columns.tolist())
    if "const" not  in exog.columns.tolist():
        print('adding constant terms to exog features')
        exog = sm.add_constant(exog)
    white_test_result = sm_diagnostic.het_white(resid=residuals,exog=exog)
    labels = ['LM statistic', 'p-value', 'f-statistic', 'f-test p-value']
    data = dict(zip(labels, white_test_result))
    
    interpretation = 'Heteroscedastic' if data['p-value'] < 0.05 else 'Homoscedastic'
    data['interpretation'] = interpretation
    result_df = pd.DataFrame(data, index=[0])
    
    return result_df


import statsmodels.api as sm
import pandas as pd

def fit_hce_regression(Y, X, cov_type='HC0',add_constant: bool = True):
    """
    Performs OLS regression with heteroscedasticity-consistent standard errors (HCE).

    Parameters:
    Y (array-like): Dependent variable
    X (pd.DataFrame): Independent variable(s)
    cov_type (str, optional): The type of heteroscedasticity-consistent covariance estimator. Default is 'HC0'.

    Returns:
    pd.DataFrame: A DataFrame containing the OLS regression results with HCE standard errors.

    Interpretation:
    The HCE method corrects the covariance matrix of the estimators to adjust the standard errors. 
    This allows hypothesis tests regarding the significance of the coefficients without problems.
    - If the p-value of a coefficient is less than the significance level (typically 0.05), 
      we reject the null hypothesis that the coefficient is equal to zero, indicating that the variable is significant.
    """
    if "const" not in X.columns.tolist() and add_constant:
        X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit().get_robustcov_results(cov_type=cov_type)
    
    summary_df = pd.DataFrame({
        'Coefficient': results.params,
        'Standard Error': results.bse,
        't-value': results.tvalues,
        'P>|t|': results.pvalues,
        '0.025': results.conf_int()[:, 0],
        '0.975': results.conf_int()[:, 1]
    })

    return results, summary_df



import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import pandas as pd

def durbin_watson_test(Y, X,add_constant: bool = False):
    """
    Conducts OLS regression and calculates the Durbin-Watson statistic to test for first-order serial autocorrelation.

    Parameters:
    Y (array-like): Dependent variable
    X (pd.DataFrame): Independent variable(s)

    Returns:
    dict: A dictionary containing the Durbin-Watson statistic and OLS regression summary.

    Interpretation:
    The Durbin-Watson test checks for first-order serial autocorrelation in the residuals of a regression.
    - Null hypothesis (H0): No first-order serial autocorrelation (DW ≈ 2).
    - Alternative hypothesis (H1): Presence of first-order serial autocorrelation (DW < 2 indicates positive autocorrelation, DW > 2 indicates negative autocorrelation).
    A DW statistic close to 2 indicates no autocorrelation. Values substantially less than 2 suggest positive autocorrelation, and values substantially greater than 2 suggest negative autocorrelation.
    """
    if "const" not in X.columns.tolist() and add_constant:
        X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    dw_statistic = durbin_watson(results.resid)

    return dw_statistic


import statsmodels.api as sm
import statsmodels.stats.diagnostic as sm_diagnostic
import pandas as pd

def breusch_godfrey_test(Y, X, nlags=1, add_constant=True):
    """
    Conducts the Breusch-Godfrey test for higher-order serial correlation.

    Parameters:
    Y (array-like): Dependent variable
    X (pd.DataFrame): Independent variable(s)
    nlags (int, optional): Number of lags to include in the test. Default is 1.
    add_constant (bool, optional): Whether to add a constant to the independent variables. Default is True.

    Returns:
    pd.DataFrame: A DataFrame containing the LM statistic, LM p-value, F-statistic, and F-test p-value.

    Interpretation:
    The Breusch-Godfrey test checks for higher-order serial correlation in the residuals of a regression.
    - Null hypothesis (H0): No serial correlation up to the specified lag.
    - Alternative hypothesis (H1): Presence of serial correlation up to the specified lag.
    If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis, indicating serial correlation.
    """

    if add_constant:
        if "const" not in X.columns:
            X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()
    bg_test = sm_diagnostic.acorr_breusch_godfrey(model, nlags=nlags)

    labels = ['LM statistic', 'LM p-value', 'F-statistic', 'F-test p-value']
    data = dict(zip(labels, bg_test))

    result_df = pd.DataFrame(data, index=[0])
    
    return result_df

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

def calculate_correlogram(residuals, nlags=30, figsize=(14, 6)):
    """
    Calculates and plots the correlogram (ACF plot) for a series of residuals.

    Parameters:
    residuals (array-like): A series of residuals
    nlags (int, optional): Number of lags to include in the ACF plot. Default is 30.
    figsize (tuple, optional): Size of the figure for the plot. Default is (14, 6).

    Returns:
    None: Displays the ACF plot.
    
    Interpretation:
    The correlogram (ACF plot) shows the autocorrelation of the residuals at different lags.
    - If the autocorrelations are within the confidence bands (usually shown as dashed lines), 
      there is no significant autocorrelation at that lag.
    - Significant spikes outside the confidence bands indicate the presence of autocorrelation 
      at the corresponding lags.
    - Patterns or gradual decay in the autocorrelations may indicate a non-random structure 
      in the residuals, suggesting model inadequacy.
    """

    fig, ax = plt.subplots(figsize=figsize)
    plot_acf(residuals, lags=nlags, zero=False, ax=ax)
    plt.show()



# %%
import statsmodels.api as sm
import pandas as pd

def ljung_box_test(residuals, lags=range(1, 6)):
    """
    Performs the Ljung-Box test for serial correlation up to a specified number of lags.

    Parameters:
    residuals (array-like): A series of residuals
    lags (range, optional): The range of lags to include in the test. Default is range(1, 6).

    Returns:
    pd.DataFrame: A DataFrame containing the Ljung-Box Q statistic and p-values for each lag.

    Interpretation:
    The Ljung-Box test checks for serial correlation in the residuals up to a specified number of lags.
    - Null hypothesis (H0): Data is distributed independently up to lag p.
    - Alternative hypothesis (H1): Data is not distributed independently up to lag p.
    If the p-value is less than the significance level (typically 0.05), we reject the null hypothesis, indicating the presence of serial correlation up to the specified lag.
    """

    ljung_box_results = sm.stats.acorr_ljungbox(residuals, lags=lags, return_df=True)
    ljung_box_results = ljung_box_results
    
    return ljung_box_results


# %%
import statsmodels.api as sm
import numpy as np
import pandas as pd

def first_difference_regression(Y, X, diff_lag=1, add_constant=True):
    """
    Performs OLS regression using first differences of the variables.

    Parameters:
    Y (array-like): Dependent variable
    X (pd.DataFrame): Independent variable(s)
    diff_lag (int, optional): The lag for differencing. Default is 1.
    add_constant (bool, optional): Whether to add a constant to the independent variables. Default is True.

    Returns:
    pd.DataFrame: A DataFrame containing the OLS regression results.

    Interpretation:
    First differencing is used to remove trends and solve autocorrelation problems in time series data.
    - If the Durbin-Watson statistic approaches the value of 2, it indicates that the autocorrelation problem has been resolved.
    - Coefficient significance, R-squared, AIC, and BIC can be used to evaluate the model fit.
    """

    Y_diff = Y.diff(diff_lag,axis=0).dropna()
    X_diff = X.diff(diff_lag,axis=0).dropna()
    
    assert Y_diff.shape[0] == X_diff.shape[0]

    if add_constant:
        X_diff = sm.add_constant(X_diff)

    model = sm.OLS(Y_diff, X_diff)
    results = model.fit()


    return results



# %%
import statsmodels.api as sm
import pandas as pd

def hac_regression(Y, X, maxlags=1, add_constant=True):
    """
    Conducts OLS regression with heteroscedasticity and autocorrelation consistent (HAC) standard errors.

    Parameters:
    Y (array-like): Dependent variable
    X (pd.DataFrame): Independent variable(s)
    maxlags (int, optional): The number of lags to include in the HAC estimator. Default is 1.
    add_constant (bool, optional): Whether to add a constant to the independent variables. Default is True.

    Returns:
    sm.regression.linear_model.RegressionResultsWrapper: The fitted model with HAC standard errors.
    pd.DataFrame: A DataFrame containing the OLS regression results with HAC standard errors.

    Interpretation:
    The HAC method corrects the covariance matrix of the estimators to adjust the standard errors, 
    allowing for valid hypothesis tests regarding the significance of the coefficients even in the presence of heteroscedasticity and autocorrelation.
    - If the p-value of a coefficient is less than the significance level (typically 0.05), 
      we reject the null hypothesis that the coefficient is equal to zero, indicating that the variable is significant.
    """

    if add_constant and "const" not in X.columns:
        X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    results = model.fit().get_robustcov_results(cov_type='HAC', maxlags=maxlags)


    return results





# %%
import statsmodels.api as sm
import pandas as pd
from statsmodels.sandbox.regression.gmm import IV2SLS

def iv_regression(Y, X, exog, endog, instruments, add_constant=True):
    """
    Conducts instrumental variables (IV) regression using 2SLS.

    Parameters:
    Y (pd.DataFrame): Dependent variable
    X (pd.DataFrame): Independent variables
    exog (list): List of exogenous feature names
    endog (list): List of endogenous feature names
    instruments (list): List of instrumental variable names
    add_constant (bool, optional): Whether to add a constant to the independent variables. Default is True.

    Returns:
    IV2SLS: The fitted 2SLS regression model object.
    """

    if add_constant:
        X = sm.add_constant(X)
        exog = exog + ["const"] 
    
    exog_data = X[exog]
    
    endog_data = X[endog]
    instrument_data = X[instruments]

    model = lm.iv.IV2SLS(Y, exog_data, endog_data, instrument_data).fit(cov_type='unadjusted')
    
    return model

def durbin_test(Y, X, exog, endog, instruments, add_constant=True):
    """
    Performs the Durbin-Wu-Hausman test for endogeneity.

    Parameters:
    Y (pd.DataFrame): Dependent variable
    X (pd.DataFrame): Independent variables
    exog (list): List of exogenous feature names
    endog (list): List of endogenous feature names
    instruments (list): List of instrumental variable names
    add_constant (bool, optional): Whether to add a constant to the independent variables. Default is True.

    Returns:
    dict: A dictionary containing the Durbin test results and interpretation.
    
    Rejecting H0 in the durbin test implies that the endogenous variables are indeed correlated with the error terms, justifying the use of IV methods.
    """

    model = iv_regression(Y, X, exog, endog, instruments, add_constant)
    durbin_result = model.durbin()
    interpretation = "Reject H0: Endogenous variables are correlated with the errors" if durbin_result.pval < 0.05 else "Fail to reject H0: Endogenous variables are exogenous"
    result = {
    'Statistic': durbin_result.stat,
    'P-value': durbin_result.pval,
    'Interpretation': interpretation
    }

    return result

def hausman_test(Y, X, exog, endog, instruments, add_constant=True):
    """
    Performs the Hausman test for endogeneity.

    Parameters:
    Y (pd.DataFrame): Dependent variable
    X (pd.DataFrame): Independent variables
    exog (list): List of exogenous feature names
    endog (list): List of endogenous feature names
    instruments (list): List of instrumental variable names
    add_constant (bool, optional): Whether to add a constant to the independent variables. Default is True.

    Returns:
    dict: A dictionary containing the Hausman test results and interpretation.
    
    Rejecting H0 in the Hausman test implies that the endogenous variables are indeed correlated with the error terms, justifying the use of IV methods.
    """

    model = iv_regression(Y, X, exog, endog, instruments, add_constant)
    hausman_result = model.wu_hausman()
    interpretation = "Reject H0: Endogenous variables are correlated with the errors" if hausman_result.pval < 0.05 else "Fail to reject H0: Endogenous variables are exogenous"
    
    result = {
        'Statistic': hausman_result.stat,
        'P-value': hausman_result.pval,
        'Interpretation': interpretation
    }
    
    return result


# %%
import statsmodels.api as sm
from linearmodels.system import SUR

def fit_sur_model(Y, X, endog, exog, add_constant=True):
    """
    Fits a Seemingly Unrelated Regression (SUR) model to a set of equations, each representing a different dependent variable with potentially correlated errors.

    Parameters:
    Y (pd.DataFrame): DataFrame containing the dependent variables for each equation.
    X (pd.DataFrame): DataFrame containing the independent variables.
    endog (list): List of column names from Y that are the dependent variables in each equation.
    exog (dict): Dictionary where each key corresponds to an equation (as listed in 'endog') and each value is a list of column names from X that are the predictors in that equation.
    add_constant (bool): If True, a constant term is added to the predictors in each equation. Default is True.

    Returns:
    sur_fit (linearmodels.system.results.SURResults): Fitted SUR model object which includes model coefficients, p-values, and other diagnostics.

    Example:
    sur_model = fit_sur_model(Y, X, endog, exog, add_constant=True)
    """
    equations = {}
    for dep_var in endog:
        specific_exog = X[exog[dep_var]]
        if add_constant:
            specific_exog = sm.add_constant(specific_exog)
        equations[dep_var] = {'dependent': Y[dep_var], 'exog': specific_exog}
    
    sur = SUR(equations)
    sur_fit = sur.fit()
    return sur_fit


# %%
import statsmodels.api as sm
import pandas as pd
from statsmodels.regression.linear_model import OLS

def compare_sur_ols(Y, X, endog, exog, add_constant=True):
    """
    Fits SUR and OLS models to the same dataset and compares their results, including R-squared values, coefficients, and p-values.

    Parameters:
    Y (pd.DataFrame): DataFrame containing the dependent variables.
    X (pd.DataFrame): DataFrame containing the independent variables.
    endog (list): List of names of the dependent variables.
    exog (dict): Dictionary where keys are dependent variable names and values are lists of independent variable names used in each respective model.
    add_constant (bool, optional): Whether to add a constant to the predictors in each model. Default is True.

    Returns:
    tuple: Contains three elements:
        - r_squared_df (pd.DataFrame): DataFrame containing the R-squared values for each model.
        - coefficients_df (pd.DataFrame): DataFrame comparing the coefficients and p-values from both OLS and SUR models for each predictor in each equation.
        - residuals_correlation (pd.DataFrame): Correlation matrix of the residuals from the OLS models.

    Example:
    rsquared_df, coefficients_df, residuals_corr = compare_sur_ols(Y, X, endog, exog, add_constant=True)
    """
    sur_model = fit_sur_model(Y, X, endog, exog, add_constant)

    ols_results = {}
    coefficients_comparison = []
    residuals = {}
    for eq in endog:
        specific_exog = X[exog[eq]]
        if add_constant:
            specific_exog = sm.add_constant(specific_exog)

        model = OLS(Y[eq], specific_exog).fit()
        ols_results[eq] = model
        residuals[eq] = model.resid

        for var in exog[eq]:
            coefficients_comparison.append({
                'Variable': var,
                'Target': eq,
                'OLS_Coefficient': model.params[var],
                'OLS_p-value': model.pvalues[var],
                'SUR_Coefficient': sur_model.params.loc[f"{eq}_{var}"],
                'SUR_p-value': sur_model.pvalues.loc[f"{eq}_{var}"]
            })

    coefficients_df = pd.DataFrame(coefficients_comparison)
    residuals_df = pd.DataFrame(residuals)
    residuals_correlation = residuals_df.corr()

    rsquared_sur = {"SUR": sur_model.rsquared}
    ols_r_squared = {f"OLS_{eq}": ols_results[eq].rsquared for eq in endog}
    merged_r_squared = rsquared_sur.copy() 
    merged_r_squared.update(ols_r_squared) 
    r_squared_df = pd.DataFrame(list(merged_r_squared.items()), columns=['Model', 'R-squared'])

    return r_squared_df, coefficients_df, residuals_correlation


# %%
import pandas as pd
import numpy as np
from scipy.stats import chi2
from linearmodels.system import SUR
import statsmodels.api as sm

def breusch_pagan_sur_test(Y, X, endog, exog, add_constant=True):
    """
    Performs the Breusch-Pagan test to check for correlation among the residuals of the equations in a SUR model.
    
    Parameters:
    Y (pd.DataFrame): DataFrame containing the dependent variables for each equation.
    X (pd.DataFrame): DataFrame containing the independent variables.
    endog (list): List of column names from Y that are the dependent variables in each equation.
    exog (dict): Dictionary where each key corresponds to an equation and each value is a list of column names from X that are the predictors in that equation.
    add_constant (bool): If True, a constant term is added to the predictors in each equation. Default is True.

    Returns:
    pd.DataFrame: DataFrame containing the Chi-squared statistic, p-value, and interpretation of the Breusch-Pagan test.
    """
    equations = {}
    if add_constant:
        X = sm.add_constant(X)

    for dep_var in endog:
        specific_exog = X[exog[dep_var]]
        equations[dep_var] = {'dependent': Y[dep_var], 'exog': specific_exog}
    
    sur = SUR(equations)
    sur_fit = sur.fit()
    
    # Extract residuals
    residuals = sur_fit.resids

    # Compute correlation matrix of residuals
    resid_corr = residuals.corr()
    n = len(Y)  # number of observations
    M = len(endog)  # number of equations

    # Breusch-Pagan statistic calculation
    lambda_bp = n * (resid_corr ** 2).sum().sum() - n * M
    df = (M * (M - 1)) / 2  # degrees of freedom
    chi2_stat = lambda_bp
    p_value = 1 - chi2.cdf(chi2_stat, df)

    # Interpretation
    interpretation = "Uncorrelated errors" if p_value > 0.05 else "Significant correlation"

    results = pd.DataFrame({
        "Chi-squared Statistic": [chi2_stat],
        "p-value": [p_value],
        "Interpretation": [interpretation]
    })

    return results



import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import Probit
import statsmodels.api as sm

def fit_probit_model(X, Y, add_constant=True):
    """
    Fits a Probit model to the given data.

    Parameters:
    X (DataFrame): The input features.
    Y (Series): The target variable, which will be binarized.
    add_constant (bool): Whether to add a constant term to the model. Default is True.

    Returns:
    result (ProbitResults): The fitted Probit model results.
    """
    # Binarize Y
    Y = (Y > 0).astype(int)
    
    # Standard scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled.index = X.index 
    if add_constant:
        X_scaled = sm.add_constant(X_scaled)
    
    # Fit the Probit model
    probit_model = Probit(Y, X_scaled)
    result = probit_model.fit()
    
    return result, Y

from statsmodels.discrete.discrete_model import MNLogit

def fit_multinomial_logit_model(Y,X, add_constant=True):
    """
    Fits a Multinomial Logistic Regression model to the given data.

    Parameters:
    X (DataFrame): The input features.
    Y (Series): The target variable with multiple categories.
    add_constant (bool): Whether to add a constant term to the model. Default is True.

    Returns:
    result (MNLogitResults): The fitted Multinomial Logit model results.
    """
    # Standard scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled.index = X.index 
    if add_constant:
        X_scaled = sm.add_constant(X_scaled)
    
    # Fit the Multinomial Logit model
    multinomial_logit_model = MNLogit(Y, X_scaled)
    result = multinomial_logit_model.fit()
    
    return result


def predict_multinomial_logit(results,X_test):
    
    if 'const' in results.params.index.tolist() and 'const' not in (X_test).columns.tolist():
    
        X_test = sm.add_constant(X_test)
    
    pred = results.predict(X_test).idxmax(axis=1)
    pred = pd.DataFrame(pred,columns = ['predictions'])
    return pred
    
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

def fit_logit_model(Y,X, add_constant=True):
    """
    Fits a Logit model to the given data.

    Parameters:
    X (DataFrame): The input features.
    Y (Series): The target variable, which will be binarized.
    add_constant (bool): Whether to add a constant term to the model. Default is True.

    Returns:
    result (LogitResults): The fitted Logit model results.
    logit_mfx (DiscreteMargins): The marginal effects of the fitted Logit model.

    Mathematical Summary:
    ---------------------
    The Logit model assumes that a binary endogenous variable Y ∈ {0,1} is influenced 
    by a set of exogenous variables X. The probability that Y = 1 follows the logistic 
    function:
    
    Pr(Y = 1 | X) = e^(Xβ) / (1 + e^(Xβ)) = Λ(Xβ)

    where Λ(·) is the cumulative distribution function (CDF) of the logistic distribution. 
    The Logit model can be expressed in terms of the odds ratio:
    
    logit(Pr(Y = 1 | X)) = ln(Pr(Y = 1 | X) / (1 - Pr(Y = 1 | X))) = Xβ

    The coefficient β_i represents the additive effect on the logarithm of the odds ratio 
    by increasing the exogenous variable X_i by a unit. If X_i is binary, it represents 
    the effect of the existence of this feature on the logarithm of the odds ratio.

    Marginal Effects:
    -----------------
    The marginal effects of the Logit model can be calculated as:
    
    ∂E(Y | X) / ∂X = (e^(Xβ) / (1 + e^(Xβ))^2) = Λ(Xβ) [1 - Λ(Xβ)] β

    These effects show the change in the probability of Y = 1 with respect to a one-unit 
    change in each of the independent variables.

    Example usage:
    --------------
    X = your feature dataframe
    Y = your target series
    result, logit_mfx = fit_logit_model(X, Y)
    print(result.summary())
    print(logit_mfx.summary())
    """
    # Binarize Y
    Y = (Y > 0).astype(int)
    
    # Standard scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled.index = X.index 
    if add_constant:
        X_scaled = sm.add_constant(X_scaled)
    
    # Fit the Logit model
    logit_model = Logit(Y, X_scaled)
    result = logit_model.fit()
    
    # Calculate marginal effects
    logit_mfx = result.get_margeff()
    
    return result, Y


from sklearn.metrics import log_loss, accuracy_score, roc_auc_score,auc,roc_curve

def plot_roc_auc(Y_true, Y_pred):
    """
    Plots the ROC AUC curve.

    Parameters:
    Y_true (Series): The true target values.
    Y_pred (Series): The predicted target values.
    """
    fpr, tpr, _ = roc_curve(Y_true, Y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



def calculate_performance_metrics(Y_true, Y_pred):
    """
    Calculates log loss, accuracy score, ROC AUC score, TPR, TNR, FPR, and FNR.

    Parameters:
    Y_true (Series): The true target values.
    Y_pred (Series): The predicted target values.

    Returns:
    metrics_df (DataFrame): DataFrame with log loss, accuracy score, ROC AUC score, TPR, TNR, FPR, and FNR.
    """
    Y_pred_rounded = np.round(Y_pred).astype(int)
    
    # Calculate confusion matrix
    cnf_matrix = confusion_matrix(Y_true, Y_pred_rounded)
    tn, fp, fn, tp = cnf_matrix.ravel()
    
    # Calculate metrics
    metrics = {
        'log_loss': log_loss(Y_true, Y_pred),
        'accuracy_score': accuracy_score(Y_true, Y_pred_rounded),
        'roc_auc_score': roc_auc_score(Y_true, Y_pred),
        'true_positive_rate (TPR)': tp / (tp + fn),
        'true_negative_rate (TNR)': tn / (tn + fp),
        'false_positive_rate (FPR)': fp / (fp + tn),
        'false_negative_rate (FNR)': fn / (fn + tp)
    }
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    return metrics_df

import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_confusion_matrix(Y_true, Y_pred):
    """
    Calculates the confusion matrix as percentages.

    Parameters:
    Y_true (Series): The true target values.
    Y_pred (Series): The predicted target values.

    Returns:
    cnf_matrix_norm (DataFrame): The normalized confusion matrix.
    """
    Y_pred_rounded = np.round(Y_pred).astype(int)
    cnf_matrix = confusion_matrix(Y_true, Y_pred_rounded)
    cnf_matrix_norm = cnf_matrix / cnf_matrix.sum(axis=1)[:, np.newaxis]
    cnf_matrix_df = pd.DataFrame(cnf_matrix_norm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
    return cnf_matrix_df

import pandas as pd
import statsmodels.api as sm

def predict_model(model, X_test, add_constant=True):
    """
    Predicts the target variable using the given model and test data.

    Parameters:
    model: The fitted Probit or Logit model.
    X_test (DataFrame): The test features.
    add_constant (bool): Whether to add a constant term to the test data. Default is True.

    Returns:
    Y_pred (DataFrame): The predicted probabilities.
    """
    if add_constant:
        X_test = sm.add_constant(X_test)
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, index=X_test.index, columns=['PRED'])
    return Y_pred_df




import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

def iv_probit_regression(Y, X, exog, endog, instruments, add_constant=True):
    """
    Conducts instrumental variables (IV) regression using a Probit model.

    Parameters:
    Y (pd.Series): Dependent variable
    X (pd.DataFrame): Independent variables
    exog (list): List of exogenous feature names
    endog (list): List of endogenous feature names
    instruments (list): List of instrumental variable names
    add_constant (bool, optional): Whether to add a constant to the independent variables. Default is True.

    Returns:
    ProbitResults: The fitted Probit regression model object.
    """
    # Binarize Y
    Y = (Y > 0).astype(int)
    
    # Standard scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled.index = X.index
    
    if add_constant:
        X_scaled = sm.add_constant(X_scaled)
        exog = exog + ["const"]
    
    # First stage: Regress the endogenous variable on the instruments and exogenous variables
    first_stage = sm.OLS(X_scaled[endog], X_scaled[instruments + exog]).fit()
    X_scaled['predicted_endog'] = first_stage.predict(X_scaled[instruments + exog])
    
    # Second stage: Probit regression using the predicted values from the first stage
    second_stage = sm.Probit(Y, X_scaled[exog + ['predicted_endog']]).fit()
    
    return second_stage



def iv_logit_regression(Y, X, exog, endog, instruments, add_constant=True):
    """
    Conducts instrumental variables (IV) regression using a Logit model.

    Parameters:
    Y (pd.Series): Dependent variable
    X (pd.DataFrame): Independent variables
    exog (list): List of exogenous feature names
    endog (list): List of endogenous feature names
    instruments (list): List of instrumental variable names
    add_constant (bool, optional): Whether to add a constant to the independent variables. Default is True.

    Returns:
    LogitResults: The fitted Logit regression model object.
    """
    # Binarize Y
    Y = (Y > 0).astype(int)
    
    # Standard scale the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled.index = X.index
    
    if add_constant:
        X_scaled = sm.add_constant(X_scaled)
        exog = exog + ["const"]
    
    # First stage: Regress the endogenous variable on the instruments and exogenous variables
    first_stage = sm.OLS(X_scaled[endog], X_scaled[instruments + exog]).fit()
    X_scaled['predicted_endog'] = first_stage.predict(X_scaled[instruments + exog])
    
    # Second stage: Logit regression using the predicted values from the first stage
    second_stage = sm.Logit(Y, X_scaled[exog + ['predicted_endog']]).fit()
    
    return second_stage



def fit_pooled_ols(Y, X, add_constant=True):
    """
    Fits a Pooled OLS model to the given panel data.

    Parameters:
    Y (pd.Series): Dependent variable with a MultiIndex [entity, time].
    X (pd.DataFrame): Independent variables with a MultiIndex [entity, time].
    add_constant (bool): Whether to add a constant term to the model. Default is True.

    Returns:
    PooledOLSResults: The fitted Pooled OLS model.
    """
    # Ensure the indices of Y and X match
    if not (Y.index.equals(X.index)):
        raise ValueError("The indices of Y and X must match and be MultiIndex.")
    
    # Optionally add a constant term
    if add_constant:
        X = sm.add_constant(X)
    
    # Fit the Pooled OLS model
    pooled_ols_model = lm.panel.PooledOLS(Y, X).fit()
    
    return pooled_ols_model


from linearmodels.panel import FamaMacBeth
def fit_fama_macbeth(Y, X, add_constant=True):
    """
    Fits a Fama-MacBeth model to the given panel data.

    Parameters:
    Y (pd.Series): Dependent variable with a MultiIndexentity * time.
    X (pd.DataFrame): Independent variables with a MultiIndex entity * time.
    add_constant (bool): Whether to add a constant term to the model. Default is True.

    Returns:
    FamaMacBethResults: The fitted Fama-MacBeth model results.
    """
    # Ensure the indices of Y and X match
    if not (Y.index.equals(X.index)):
        raise ValueError("The indices of Y and X must match and be MultiIndex.")
    
    # Optionally add a constant term
    if add_constant:
        X = sm.add_constant(X)
    
    # Fit the Fama-MacBeth model
    model = FamaMacBeth(Y, X)
    results = model.fit()
    return results
    
    

from linearmodels.panel import BetweenOLS
import statsmodels.api as sm
import pandas as pd

def fit_between_ols(Y, X, add_constant=True):
    """
    Fits a Between OLS model to the given panel data.

    The Between OLS (Between Effects) estimator is used to analyze panel data by focusing on cross-sectional variation between entities.
    This method averages the data over time for each entity and then performs a cross-sectional regression on these averages.

    Parameters:
    Y (pd.Series): Dependent variable with a MultiIndex (entity * time).
    X (pd.DataFrame): Independent variables with a MultiIndex (entity * time).
    add_constant (bool): Whether to add a constant term to the model. Default is True.

    Returns:
    BetweenOLSResults: The fitted Between OLS model results.

    What Between OLS Does:
    - **Averaging Over Time**: For each entity, the dependent variable (Y) and the independent variables (X) are averaged over time.
      \[
      \bar{Y}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} Y_{it}
      \]
      \[
      \bar{X}_i = \frac{1}{T_i} \sum_{t=1}^{T_i} X_{it}
      \]
      where \( T_i \) is the number of time periods for entity \( i \).
    - **Cross-Sectional Regression**: Perform an OLS regression using these time-averaged values.
      \[
      \bar{Y}_i = \alpha + \beta \bar{X}_i + \bar{\mu}_i + \bar{\epsilon}_i
      \]
      Here, \( \bar{Y}_i \) and \( \bar{X}_i \) are the time-averaged dependent and independent variables for entity \( i \),
      \( \alpha \) is the intercept, \( \beta \) is the coefficient vector, \( \bar{\mu}_i \) captures entity-specific effects,
      and \( \bar{\epsilon}_i \) is the error term.

    Steps:
    1. Ensure the indices of Y and X match.
    2. Optionally add a constant term to X.
    3. Fit the Between OLS model using the `BetweenOLS` class from the `linearmodels` package.

    Example:
    ```
    import pandas as pd
    from linearmodels.panel import BetweenOLS

    # Example DataFrame df with panel data
    df = pd.read_csv('panel_data.csv')
    Y = df['Y']
    X = df[['X1', 'X2']]

    # Ensure Y and X have a MultiIndex (entity * time)
    Y.index = pd.MultiIndex.from_arrays([df['entity'], df['time']])
    X.index = pd.MultiIndex.from_arrays([df['entity'], df['time']])

    results = fit_between_ols(Y, X)
    print(results.summary)
    ```
    """
    # Ensure the indices of Y and X match
    if not (Y.index.equals(X.index)):
        raise ValueError("The indices of Y and X must match and be MultiIndex.")
    
    # Optionally add a constant term
    if add_constant:
        X = sm.add_constant(X)
    
    # Fit the Between OLS model
    model = BetweenOLS(Y, X,check_rank=False)
    results = model.fit()
    
    return results



from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import pandas as pd
class RegressionPanelOLS:
    
    def __init__(self,Y, X, entity_effect:bool = False,
                 time_effect:bool = False,
                 other_effect: list= None,
                 add_constant : bool = True):
        
        
        self.Y = Y
        self.X = X
        self.entity_effect = entity_effect
        self.time_effect = time_effect
        self.other_effect = other_effect
        self.add_constant = add_constant
        
    
    def fit(self):
        """
        Fits a Panel OLS model to the given panel data, with optional entity, time, or other effects.

        Parameters:
        Y (pd.Series): Dependent variable with a MultiIndex (entity * time).
        X (pd.DataFrame): Independent variables with a MultiIndex (entity * time).
        add_constant (bool): Whether to add a constant term to the model. Default is True.
        entity_effects (bool): Flag to include entity (fixed) effects in the model. Default is False.
        time_effects (bool): Flag to include time effects in the model. Default is False.
        other_effects (list of str | None): List of column names to use for any other effects that are not entity or time effects. Default is None.

        Returns:
        results: The fitted Panel OLS model results.
        estimated_effects_df: DataFrame with estimated effects for the other effects (only if other_effects is not None).

        Notes:
        - **Entity Effects**: Control for unobserved heterogeneity across entities that is constant over time.
        \[
        Y_{it} = \alpha + \beta X_{it} + \mu_i + \epsilon_{it}
        \]
        where \( \mu_i \) are the entity-specific effects.

        - **Time Effects**: Control for time-specific shocks or trends that affect all entities equally.
        \[
        Y_{it} = \alpha + \beta X_{it} + \lambda_t + \epsilon_{it}
        \]
        where \( \lambda_t \) are the time-specific effects.

        - **Both Entity and Time Effects**: Control for both unobserved heterogeneity across entities and time-specific shocks.
        \[
        Y_{it} = \alpha + \beta X_{it} + \mu_i + \lambda_t + \epsilon_{it}
        \]

        - **Other Effects**: Control for additional categorical dimensions.
        \[
        Y_{it} = \alpha + \beta X_{it} + \gamma Z_{i} + \epsilon_{it}
        \]
        where \( Z_{i} \) represents other categorical effects.

        - If both entity_effects and time_effects are False, and no other effects are included, the model reduces to Pooled OLS.

        - Only two effects (entity, time, or other) can be used simultaneously.

        Raises:
        ValueError: If more than two effects are enabled or if indices of Y and X do not match.

        Example:
        ```
        import pandas as pd
        from linearmodels.panel import PanelOLS

        # Example DataFrame df with panel data
        df = pd.read_csv('panel_data.csv')
        Y = df['Y']
        X = df[['X1', 'X2']]

        # Ensure Y and X have a MultiIndex (entity * time)
        Y.index = pd.MultiIndex.from_arrays([df['entity'], df['time']])
        X.index = pd.MultiIndex.from_arrays([df['entity'], df['time']])

        results, estimated_effects_df = fit_panel_ols(Y, X, entity_effects=True, time_effects=True, other_effects=['sector'])
        print(results.summary)
        print(estimated_effects_df)
        ```
        """
        # Ensure the indices of Y and X match
        if not (self.Y.index.equals(self.X.index)):
            raise ValueError("The indices of Y and X must match and be MultiIndex.")

        # Check that not more than two effects are enabled
        effects = [self.entity_effect, self.time_effect, self.other_effect is not None]
        if sum(effects) > 2:
            raise ValueError("Only two effects (entity, time, or other) can be used simultaneously.")

        # Handle other effects
        if self.other_effect is not None:
            if len(self.other_effect) > 2:
                raise ValueError("Only two other effects can be used simultaneously.")
        other_effects_data = self.X[self.other_effect]
        X = self.X.drop(columns=self.other_effect)
        self.features = X.columns.tolist()
        # Optionally add a constant term
        if self.add_constant:
            X = sm.add_constant(X)

        # Fit the appropriate Panel OLS model
        if self.other_effect is not None:
            self.model = PanelOLS(dependent=self.Y, exog=X,
                                  entity_effects=self.entity_effect,
                                  time_effects=self.time_effect,
                                  other_effects=other_effects_data,
                                  drop_absorbed=True)
        else:
            self.model = PanelOLS(dependent=self.Y, exog=X,
                                  entity_effects=self.entity_effect,
                                  time_effects=self.time_effect,
                                  drop_absorbed=True)

        self.results = self.model.fit(cov_type="robust")
        
        
        self.compute_marginal_effect()
        
    
    def compute_marginal_effect(self):
        eff = self.results.estimated_effects
        eff = np.round(eff,3)
        eff = eff.join(self.X[self.other_effect]).drop_duplicates()
        eff = eff.set_index(self.other_effect).sort_index()
        self.eff_mapping = eff['estimated_effects'].to_dict()
        self.eff = eff
        
        
    def predict(self,X_test):
        if self.add_constant:
            X_test = sm.add_constant(X_test)
        print(X_test)
        print(self.results.params)
        self.idyosincratic = X_test[self.features + ['const']] @ self.results.params
        if self.other_effect is not None:
            effects = X_test[self.other_effect]
            effects = effects.iloc[:,0].map(self.eff_mapping)
            pred = self.idyosincratic + effects
            pred = pd.DataFrame(pred,columns = ['predictions'])
        else :

            pred = pd.DataFrame(self.idyosincratic,columns = ['predictions'])
    
        return pred
        