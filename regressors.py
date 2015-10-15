import numpy as np
__author__ = "Aidan O'Brien"


class Regressor(object):
    """
    Base class for regression objects. Contains error and evaluation functions
    """
    def __init__(self, df, features=[], target=''):
        self.df = df
        self.features = features
        self.target = target
        self.df_featured = df[features]
        self.df_target = df[target]
        self.targets = self.df_target.values
        self.featured_values = np.c_[np.ones(self.df_featured.values.shape[0]), self.df_featured.values]

    def __train__(self):
        pass

    def retrain(self, df, features=[], target=''):
        """
        Retrains the object with a new dataframe, features and target
        :param df: A pandas DataFrame
        :param features: The columns that are selected to be the features of the new regressor
        :param target: The column that contains the target of the new regressor
        :return: updates the theta value
        """
        self.df = df
        self.set_new_features(features)
        self.set_new_target(target)

    def evaluate(self, df=None):
        """
        Evaluates the errors between the predicted values and a test set
        :param df: A DataFrame of the same format as the training set
        :return: returns the maximum error and the RMSE in a dict format
        """
        if df is None:
            df = self.df
        rss = self.__calculate_residual_sum_of_squares__(df)
        rmse = self.__calculate_residual_mean_squared_error__(df)
        max_error = self.__calculate_max_error__(df)
        return {'max_error': max_error,
                'rmse': rmse,
                'rss': rss}

    def __calculate_residual_mean_squared_error__(self, df):
        """
        Calculates the RMSE for evaluations for the given DataFrame
        :return:
        """
        y_pred = np.c_[np.ones(df[self.features].values.shape[0]), df[self.features].values].dot(self.theta)
        return (np.sum((df[self.target].values-y_pred)**2) / df[self.target].values.shape[0])**0.5

    def __calculate_residual_sum_of_squares__(self, df):
        """
        Calculates the residual sum of squares for a trained model
        :return: updates the rss attribute
        """
        y_pred = np.c_[np.ones(self.df_featured.values.shape[0]), self.df_featured.values].dot(self.theta)
        return 1 - np.sum((self.targets - y_pred)**2) / np.sum((self.targets - np.mean(self.targets))**2)

    def __calculate_max_error__(self, df):
        """
        Finds the maximum error between a set of predictions and a set of known outputs
        :param df: A DataFrame of the same form as the training set
        :return: The absolute error of an individual target and prediction
        """
        y_pred = np.c_[np.ones(df[self.features].values.shape[0]), df[self.features].values].dot(self.theta)
        return np.max(np.abs(df[self.target].values - y_pred))

    def set_new_features(self, features=[]):
        self.features = features
        self.df_featured = self.df[features]
        self.featured_values = np.c_[np.ones(self.df_featured.values.shape[0]), self.df_featured.values]

    def set_new_target(self, target=''):
        self.target = target
        self.df_target = self.df[target]
        self.targets = self.df_target.values

    def get_features(self):
        return self.features

    def get_target(self):
        return self.target
