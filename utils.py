import pandas as pd 
import numpy as np


def rolling_correlation(big_df, single_col_df, window):
    # Ensure single_col_df is a Series
    single_col_series = single_col_df.squeeze()

    # Define a function to calculate rolling correlation for a single column
    def rolling_corr(col):
        return col.rolling(window).corr(single_col_series)
    
    # Apply the rolling correlation function to each column
    result_df = big_df.apply(rolling_corr, axis=0)
    
    return result_df

def map_values(big_df, single_col_df):
    # Reindex single_col_df to match the index of big_df
    single_col_df_reindexed = single_col_df.reindex(big_df.index).ffill()
    
    # Broadcast the single column values to match the shape of big_df
    new_df = big_df.apply(lambda x: single_col_df_reindexed.iloc[:, 0])
    
    return new_df


def rolling_z_score(df, window):
    rolling_mean = df.rolling(window=window, center=True).mean()
    rolling_std = df.rolling(window=window, center=True).std()
    z_scores = (df - rolling_mean) / rolling_std
    return z_scores



def change_to_first_business_day(df, next=False):
    if next:
        # Change index to the first business day of the following month
        new_index = df.index + pd.offsets.BMonthBegin(1)
    else:
        # Change index to the first business day of the current month
        new_index = df.index.to_period('M').to_timestamp() + pd.offsets.BMonthBegin(0)
    
    df.index = new_index
    return df


def expanding_window_qcut(df, column, q):
    # Creating an empty DataFrame to store the results
    result = pd.DataFrame(index=df.index, columns=[column + '_qcut'])
    
    for i in range(1, len(df) + 1):
        result.iloc[i - 1] = pd.qcut(df[column].iloc[:i], q, labels=False, duplicates='drop')[-1]
    
    return result

def expanding_window_qcut_no_loop(df, column, q, min_periods=1):
    # Initialize an array to hold the qcut labels
    qcut_labels = np.full(len(df), np.nan)

    for i in range(min_periods, len(df) + 1):
        current_ranks = df[column].iloc[:i].rank(method='average', pct=True)
        try:
            qcut_labels[i - 1] = pd.qcut(current_ranks, q, labels=False, duplicates='drop')[-1]
        except ValueError:
            qcut_labels[i - 1] = np.nan

    # Fill NaN values with -1 to indicate insufficient data
    qcut_labels = pd.Series(qcut_labels, index=df.index).fillna(-1).astype(int)

    # Return the result as a DataFrame
    result = pd.DataFrame(qcut_labels, columns=[column + '_qcut'])
    return result

def row_qcut(df, q, ascending=True):
    def qcut_row(row):
        valid_values = row.dropna()
        if len(valid_values) == 0:
            return row
        quantiles = pd.qcut(valid_values, q, labels=False, duplicates='drop')
        if not ascending:
            quantiles = q - 1 - quantiles
        row.loc[valid_values.index] = quantiles + 1  # Adding 1 to make it 1-based index
        return row
    
    return df.apply(qcut_row, axis=1) 
    
    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def center(x):
    mean = x.mean(1)
    return x.sub(mean,0)



def plot_true_vs_pred(df, pred_column):
    """
    Plots a scatter plot of the true data vs. predicted data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the true and predicted data.
    pred_column (str): The name of the column containing the predicted data. The other column will be assumed to contain the true data.

    Returns:
    None: Displays a scatter plot.

    Example:
    df = pd.DataFrame({
        'Y_true': [1, 2, 3, 4, 5],
        'Y_pred': [1.1, 1.9, 3.05, 4.2, 4.8]
    })
    plot_true_vs_pred(df, 'Y_pred')
    """
    # Extract true and predicted values
    true_column = [col for col in df.columns if col != pred_column][0]
    Y_true = df[true_column]
    Y_pred = df[pred_column]

    # Calculate R-squared
    correlation_matrix = np.corrcoef(Y_true, Y_pred)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy**2

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_true, Y_pred, alpha=0.7, label=f'Y_pred vs {true_column}')
    plt.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--', label='Y_true = Y_pred')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.title(f'True vs Predicted Values (R^2 = {r_squared:.4f})')
    plt.grid(True)
    plt.show()



def center_and_standardize(df, features, level, standardize=False):
    def center_and_std(group):
        centered = group[features] - group[features].mean()
        if standardize:
            centered = centered / group[features].std()
        return centered

    centered_df = df.groupby(level=level, group_keys=False).apply(center_and_std)
    for feature in features:
        df[feature] = centered_df[feature]
    
    return df




def build_two_factor_portfolio(factor1,factor2,q ):
    
    factor1_qcut = row_qcut(factor1,q=q,ascending=True)
    first_filter_long = (factor1_qcut==q)
    first_filter_short = (factor1_qcut==1)
    second_filter_long = row_qcut(factor2[first_filter_long],q= q,ascending=True)
    second_filter_short =  row_qcut(factor2[first_filter_short],q= q,ascending=False)
    
    long_leg = second_filter_long[second_filter_long==q] / q
    short_leg = second_filter_short[second_filter_short==1] *-1
    portfolio = pd.concat([short_leg.stack(),long_leg.stack()],axis=0).sort_index(level=0)
    
    return portfolio

def build_two_factor_portfolio(factor1,factor2,pos_filter_1,pos_filter_2):

    factor1_long = factor1.rank(axis=1,ascending=False,pct=False)
    first_filter_long = (factor1_long<=pos_filter_1)
    factor1_short = factor1.rank(axis=1,ascending=True,pct=False)
    first_filter_short = (factor1_short<=pos_filter_1)

    second_filter_long = factor2[first_filter_long].rank(axis=1,ascending=False,pct=False)
    second_filter_short = factor2[first_filter_short].rank(axis=1,ascending=True,pct=False)

    long_leg = second_filter_long<=pos_filter_2
    long_leg = long_leg.replace({True:1,False:np.nan})

    short_leg = second_filter_short<=pos_filter_2
    short_leg = short_leg.replace({True:-1,False:np.nan})

    portfolio = pd.concat([short_leg.stack(),long_leg.stack()],axis=0).sort_index(level=0)

    return portfolio