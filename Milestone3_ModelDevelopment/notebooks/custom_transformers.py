import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from verstack import NaNImputer

############################################################

class DataCleaner(BaseEstimator, TransformerMixin):
	def __init__(self, cols_to_drop, nonnegative_cols):
		self.cols_to_drop = cols_to_drop
		self.nonnegative_cols = nonnegative_cols
   
	def fit(self, X, y=None):
		return self

	# X is pd.DataFrame
	def transform(self, X):
		X_copy = X.copy()
		X_copy.drop(columns=self.cols_to_drop, errors='ignore', inplace=True)	
			
		X_copy.replace(['?', 'Error'], np.nan, inplace=True)
		
		if 'avg_frequency_login_days' in X_copy.columns:
			X_copy['avg_frequency_login_days'] = X_copy['avg_frequency_login_days'].astype(float)

		for col in self.nonnegative_cols:
			if col in X_copy.columns:
				X_copy.loc[X_copy[col] < 0, col] = np.nan

		return X_copy
	
############################################################

class NaNImputerWrapper(BaseEstimator, TransformerMixin):
	def __init__(self, train_sample_size=30_000, verbose=True):
		self.train_sample_size = train_sample_size
		self.verbose = verbose
		self.imputer = NaNImputer(self.train_sample_size, self.verbose)

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		return self.imputer.impute(X)

############################################################

class FeatureEng(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.membership_order = ['No Membership', 'Basic Membership', 'Silver Membership',
								 'Gold Membership', 'Platinum Membership', 'Premium Membership']
		self.positive_feedback = ['Products always in Stock', 'Quality Customer Care', 'Reasonable Price', 'User Friendly Website']
		self.negative_feedback = ['Poor Website', 'Poor Customer Service', 'Poor Product Quality', 'Too many ads']

	def get_sentiment(self, feedback):
		if feedback in self.positive_feedback:
			return 1
		elif feedback in self.negative_feedback:
			return -1
		else:
			return 0

	def fit(self, X, y=None):
		return self

	def transform(self, X):
        # feature selection
		X = X[['membership_category', 'feedback', 'points_in_wallet']]

        # encoding
		X['membership_category'] = pd.Categorical( X['membership_category'], 
												  categories=self.membership_order, 
												  ordered=True).codes
		
		X['feedback'] = X['feedback'].apply(self.get_sentiment)

		# standardization
		X['points_in_wallet'] = (X['points_in_wallet'] - X['points_in_wallet'].mean()) / X['points_in_wallet'].std()
		
		return X  
	
	def fit_transform(self, X, y=None):
		X_transformed = self.transform(X)
		self.feature_names_out_ = X_transformed.columns
		return X_transformed
	
	def get_feature_names_out(self, input_features=None):
		return self.feature_names_out_

############################################################