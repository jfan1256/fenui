import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import ray
import re

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class Model:
    def __init__(self,
                 data: pd.DataFrame = None,
                 beta_col: [] = None,
                 d_to_m: bool = None,
                 y_col: str = None,
                 split: int = None,
                 epoch: int = None,
                 batch_size: int = None,
                 alpha: int = None,
                 alpha_tune: [] = None,
                 l1_ratio: int = None,
                 ):

        '''
        data (pd.DataFrame): Pandas Dataframe that holds the data to perform Linear Regression
        beta_col (list[str]): List of the beta columns to use as the X in Linear Regression or Neural Network
        d_to_m (bool): Aggregate daily data to monthly data after Linear Regression or not
        y_col (str): Name of column used as the Y in Linear Regression or Neural Network
        split (float): Split Ratio for Train/Test (i.e., 0.8 means 0.8 train, 0.2 test)
        epoch (int): Number of epochs to train for the Neural Network
        batch_size (int): Batch size for the Neural Network
        alpha (float): Alpha value inputted for L1, L2, and ElasticNet Regression
        alpha_tune (list[float]): List of alpha values for GridSearch Tuning
        l1_ratio (float): L1/L2 Alpha Ratio for ElasticNet Regression
        '''

        self.data = data
        self.beta_col = beta_col
        self.d_to_m = d_to_m
        self.y_col = y_col
        self.split = split
        self.epoch = epoch
        self.batch_size = batch_size
        self.alpha = alpha
        self.alpha_tune = alpha_tune
        self.l1_ratio = l1_ratio

    # Demean R2
    def demean_r2(self, predictions, Y):
        if self.d_to_m:
            predictions = pd.DataFrame({'data': predictions}, index=Y.index)
            Y = Y.resample('M').mean()
            predictions = predictions.resample('M').mean()
            sse = np.sum((Y.squeeze() - predictions.squeeze()) ** 2)
            tss = np.sum((Y.squeeze() - np.mean(Y.squeeze())) ** 2)
            r2 = 1 - (sse / tss)
        else:
            predictions = pd.DataFrame({'data': predictions}, index=Y.index)
            sse = np.sum((Y.squeeze() - predictions.squeeze()) ** 2)
            tss = np.sum((Y.squeeze() - np.mean(Y.squeeze())) ** 2)
            r2 = 1 - (sse / tss)
        return r2

    # Mean R2
    def mean_r2(self, predictions, Y):
        if self.d_to_m:
            predictions = pd.DataFrame({'data': predictions}, index=Y.index)
            Y = Y.resample('M').mean()
            predictions = predictions.resample('M').mean()
            sse = np.sum((Y.squeeze() - predictions.squeeze()) ** 2)
            tss = np.sum(Y.squeeze() ** 2)
            r2 = 1 - (sse / tss)
        else:
            predictions = pd.DataFrame({'data': predictions}, index=Y.index)
            sse = np.sum((Y.squeeze() - predictions.squeeze()) ** 2)
            tss = np.sum(Y.squeeze() ** 2)
            r2 = 1 - (sse / tss)
        return r2

    # Plot scatter with 45 degree line
    @staticmethod
    def scatter_45(x, y, x_axis, y_axis, title):
            plt.figure(figsize=(8, 8))
            plt.scatter(x, y)
            # Add labels and title
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.title(title)
            # Determine limits for equal scale
            combined = np.concatenate([x, y])
            min_val = combined.min()
            max_val = combined.max()
            # Set limits for x and y axes
            plt.xlim(min_val, max_val)
            plt.ylim(min_val, max_val)
            # Plot a 45-degree line
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            # Set aspect of plot to be equal
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
            plt.close()

    @staticmethod
    def get_max_r2(data):
        max_key = max(data, key=lambda k: data[k]['oos_r2'])
        max_all = {max_key: data[max_key]}
        return max_all

    # Plot In-sample vs. Out-of-Sample scatter
    @staticmethod
    def scatter_is_oos(x1, y1, x2, y2, x_axis, y_axis, title):
        plt.scatter(x1, y1, label='In Sample')
        plt.scatter(x2, y2, label='Out of Sample')
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend()
        plt.savefig(get_reports() / 'scatter_is_oos' / title)
        plt.close()


    # Plot time-series
    @staticmethod
    def time_series(actual_values, predictions, title):
        # Set the figure size
        plt.figure(figsize=(40, 10))
        plt.plot(actual_values, label='Actual Values')
        plt.plot(actual_values.index, predictions, label='Predicted Values')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        plt.close()

    # Split train/test
    def split_data(self):
        X = self.data[self.beta_col]
        Y = self.data[self.y_col]
        split_index = int(len(self.data) * self.split)
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        Y_train = Y.iloc[:split_index]
        Y_test = Y.iloc[split_index:]
        return X_train, X_test, Y_train, Y_test

    # Get IS and OOS predictions + R2
    def pred_r2(self, model, X_train, X_test, Y_train, Y_test):
        is_pred = model.predict(X_train)
        oos_pred = model.predict(X_test)
        is_r2 = self.demean_r2(is_pred, Y_train)
        is_dr2 = self.mean_r2(is_pred, Y_train)
        oos_r2 = self.demean_r2(oos_pred, Y_test)
        oos_dr2 = self.mean_r2(oos_pred, Y_test)
        pred_dict = {'X_train': X_train, 'Y_train': Y_train, 'Y_train': Y_train, 'Y_test': Y_test,
                     'is_pred': is_pred, 'oos_pred': oos_pred, 'is_r2': is_r2, 'oos_r2': oos_r2}
        return pred_dict

    # IS OLS
    def is_ols(self):
        # Setup Data
        X = self.data[self.beta_col]
        X = sm.add_constant(X)
        Y = self.data[self.y_col]
        # Run Model
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X)
        # Get metrics
        r2 = self.mean_r2(Y, predictions)
        dr2 = self.demean_r2(Y, predictions)
        print("Mean R2: ", r2)
        print("Demean R2: ", dr2)
        self.scatter_45(Y, predictions, 'Actual Value', 'Predicted Value', self.y_col)
        self.time_series(Y, predictions, self.y_col)
        pred_dict = {'is_actual': Y, 'is_pred': predictions, 'is_r2': r2, 'is_dr2': dr2}
        return pred_dict

    # OOS OLS
    def oos_ols(self):
        # Setup Data
        X_train, X_test, Y_train, Y_test = self.split_data()
        X_train = sm.add_constant(X_train)
        # Run Model
        model = sm.OLS(Y_train, X_train).fit()
        # Get metrics
        pred_dict = self.pred_r2(model, X_train, X_test, Y_train, Y_test)
        return pred_dict

    # OOS L1 OLS
    def oos_l1_ols(self):
        # Setup Data
        X_train, X_test, Y_train, Y_test = self.split_data()
        # Run Model
        model = Lasso(alpha=self.alpha)
        model.fit(X_train, Y_train)
        # Get metrics
        pred_dict = self.pred_r2(model, X_train, X_test, Y_train, Y_test)
        return pred_dict

    # OOS L2 OLS
    def oos_l2_ols(self):
        # Setup Data
        X_train, X_test, Y_train, Y_test = self.split_data()
        # Run Model
        model = Ridge(alpha=self.alpha)
        model.fit(X_train, Y_train)
        # Get metrics
        pred_dict = self.pred_r2(model, X_train, X_test, Y_train, Y_test)
        return pred_dict

    # OOS EN OLS
    def oos_en_ols(self):
        # Setup Data
        X_train, X_test, Y_train, Y_test = self.split_data()
        # Run Model
        model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        model.fit(X_train, Y_train)
        # Get metrics
        pred_dict = self.pred_r2(model, X_train, X_test, Y_train, Y_test)
        return pred_dict

    # OOS NN
    def oos_nn(self):
        # Setup Data
        X_train, X_test, Y_train, Y_test = self.split_data()
        # Run Model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(len(self.beta_col),)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(), loss='mean_squared_error')
        model.fit(X_train, Y_train, self.epoch, self.batch_size)
        # Get metrics
        pred_dict = self.pred_r2(model, X_train, X_test, Y_train, Y_test)
        return pred_dict

    # Tune Hyperparameters
    def tune(self, func):
        tune_dict = {}
        if isinstance(self.alpha_tune[0], list):
            for parameter in self.alpha_tune:
                self.alpha = parameter[0]
                self.l1_ratio = parameter[1]
                pred_dict = func()
                subset_dict = {key: pred_dict[key] for key in ['is_r2', 'oos_r2']}
                tune_dict[parameter] = subset_dict
            return tune_dict

        else:
            alpha_collect = []
            is_r2_collect = []
            oos_r2_collect = []
            for parameter in self.alpha_tune:
                self.alpha = parameter
                alpha_collect.append(self.alpha)
                pred_dict = func()
                subset_dict = {key: pred_dict[key] for key in ['is_r2', 'oos_r2']}
                tune_dict[parameter] = subset_dict
                is_r2_collect.append(subset_dict['is_r2'])
                oos_r2_collect.append(subset_dict['oos_r2'])

            emb_name = ''.join(re.findall("[a-zA-Z]+", self.beta_col[0]))
            title = f'{emb_name}_{self.y_col}'
            # self.scatter_is_oos(alpha_collect, is_r2_collect, alpha_collect, oos_r2_collect, 'Alpha', 'Demean R2', title)
            return tune_dict


    # Process Y columns (This will be parallelized in tune_multiple_y)
    @staticmethod
    @ray.remote
    def process_column(model, data, d_to_m, beta_col, y_col, split, alpha_tune):
        tune_model = Model(data=data, d_to_m=d_to_m, beta_col=beta_col, y_col=y_col, split=split, alpha_tune=alpha_tune)

        if model == "oos_l2_ols":
            metric_dict = tune_model.tune(tune_model.oos_l2_ols)
        elif model == "oos_l1_ols":
            metric_dict = tune_model.tune(tune_model.oos_l1_ols)
        elif model == "oos_en_ols":
            metric_dict = tune_model.tune(tune_model.oos_en_ols)
        elif model == "oos_nn":
            metric_dict = tune_model.tune(tune_model.oos_nn)

        max_r2 = tune_model.get_max_r2(metric_dict)
        return y_col, max_r2

    # Regressing on multiple Y variables
    def tune_multiple_y(self, model):
        ray.init(num_cpus=16, ignore_reinit_error=True)
        futures = [self.process_column.remote(model, self.data, self.d_to_m, self.beta_col, col, self.split, self.alpha_tune) for col in self.y_col]
        col_result = {col: max_r2 for col, max_r2 in ray.get(futures)}
        ray.shutdown()
        return col_result

    # Table for tune_multiple_y
    @staticmethod
    def table_multiple_y(data):
        rows = []
        for y_var, alpha_dict in data.items():
            for alpha, metrics in alpha_dict.items():
                row = {'Y Variable': y_var, 'alpha': alpha, **metrics}
                rows.append(row)

        df = pd.DataFrame(rows)
        df['is_r2'] = df['is_r2'].apply(lambda x: max(x, 0))
        df['oos_r2'] = df['oos_r2'].apply(lambda x: max(x, 0))
        return df

