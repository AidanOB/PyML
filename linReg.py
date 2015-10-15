from regressors import Regressor
import numpy.linalg as la

__author__ = "Aidan O'Brien"

# Linear regression for pandas DataFrames


class LinearRegressor(Regressor):
    """
    This class takes a pandas DataFrame and creates a trained model based upon the data which can then be used to
    provide predictions and can be compared to test data. This object assumes that the data has been cleaned correctly.
    """
    def __init__(self, df, features=[], target=''):
        Regressor.__init__(self, df, features, target)
        self.theta = self.__train__()
        self.rmse = self.__calculate_residual_mean_squared_error__(self.df)
        self.rss = self.__calculate_residual_sum_of_squares__(df)

    def __train__(self):
        """
        Uses the features from the DataFrame to form a model that achieves the optimum line to the target
        :return: An array with the variables for the required features
        """
        return la.solve(self.featured_values.T.dot(self.featured_values),
                        self.featured_values.T.dot(self.df_target.values))

    def retrain(self, df, features=[], target=''):
        """
        Retrains the object with a new dataframe, features and target
        :param df: A pandas DataFrame
        :param features: The columns that are selected to be the features of the new regressor
        :param target: The column that contains the target of the new regressor
        :return: updates the theta value
        """
        Regressor.retrain(self, df, features, target)
        self.theta = self.__train__()
        self.rss = self.__calculate_residual_mean_squared_error__(self.df)

    def predict_target(self, df):
        """
        Takes a DataFrame, then returns the predicted target based upon the same features.
        :param df: A new DataFrame with the same features as the training DataFrame
        :return: Predicted target for the features
        """
        return df[self.features].values.dot(self.theta)