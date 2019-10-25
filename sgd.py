#!/usr/bin/env python
# coding: utf-8

# 
# Title: SGD Regressor for Property Prediction
# =======
# - Created: 2019.10.03
# - Updated: 2019.10.11
# - Author: Kyung Min, Lee
# 
# Learned from 
# - "Chapter 2 of Hands-on Machine Learning Book"
# - Sckit-Learn documents
# 

# 

# ## SGD stands for Stochastic Gradient Descent: the gradient of the loss is estimated each sample at a time and the model is updated along the way with a decreasing strength schedule (aka learning rate).
# 
# > class sklearn.linear_model.SGDRegressor(loss=’squared_loss’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate=’invscaling’, eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)[source]

# HyperParameters
# ---
# 

# 
# -   **loss**:  : str, default: ‘squared_loss’
#   - The loss function to be used. The possible values are *‘squared_loss’, ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’*
#     - *‘squared_loss’* refers to the ordinary least squares fit(DEFAULT)
#     -  *‘huber’* modifies ‘squared_loss’ to focus less on getting outliers correct by switching from squared to linear loss past a distance of epsilon.
#     -  *‘epsilon_insensitive’* ignores errors less than epsilon and is linear past that; this is the loss function used in SVR.
#     - *‘squared_epsilon_insensitive’* is the same but becomes squared loss past a tolerance of epsilon.  
#   
#   
# - **penalty**:  str, ‘none’, ‘l2’, ‘l1’, or ‘elasticnet’
#   - The penalty (aka regularization term) to be used. 
#     - *l2*(DEFAULT): standard regularizer for linear SVM models. => <U>Lidge Regression</U>
#     $$ min(||Y-\widehat{Y}||^2_2 + \lambda||\theta||^2_2$$
#       - $\lambda$: ridge function의 alpha값, alpha값이 클 수록 penalty도 커지면서 계수의 크기가 줄어든다. 이는 변수를 축소하면서 다중공선선(multicollinearity)를 방지하는데 쓰임
#       - 모든 변수를 가지고 오면서 계수값을 줄인다. 하지만 model이 아직까지도 복잡한 상태이고 변수도 유지하므로 모델 성능 저하에 영향을 미칠 수 있다.
#       
#     - *l1*: Regularization with l1 norm, 중요한 몇 개의 변수만 선택하고 다른 계수들은 0으로 줄인다. 이 특징은 Feature selection으로 알려져 있고 ridge regression은 이 과정이 없다 => <U>Lasso Regression</U>
#       - 변수를 자동채택 할 수 있어 일반적으로 많은 변수를 다룰 때 활용
#     $$ min(||Y-\widehat{Y}||^2_2 + \lambda||\theta||^2_1$$
#       - 변수들 끼리 상관관계가 존재한다면 Lasso는 단 한개의 변수만 채택하므로 정보의 손실을 야기하여 정확성이 떨어질 수 있다.
#       
#     - *elasticnet*: Ridge & Lasso regression의 조합=> <U>Elastic Net</U>
#       $$ min(||Y-\widehat{Y}||^2_2 + \lambda_1||\theta||^2_1+ \lambda_2||\theta||^2_2$$
#       - 상관 관계가 존재하는 변수로 구성된 그룹을 형성한다. 이 그룹의 변수 중 하나가 강력한 예측변수가 있다면 이 그룹을 model building에 포함시킨다. 상관관계를 최대한 유지하면서 필요없는 변수의 영향을 최대한 지우는 것
#       
#       $ Alpha = \lambda_1 + \lambda_2 $    and       l1_ratio$  = \frac{\lambda_1 }{\lambda_1 + \lambda_2} $
#       
#       
# - **alpha**[float]: Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.  &nbsp;
# 
# - **1l_ratio**[float]: The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1. l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1. Defaults to 0.15.  
# 
# - **fit_intercept**[bool]: Whether the intercept should be estimated or not. If False, the data is assumed to be already centered. Defaults to True.  
# 
# - **max_iter**[int, optional](defalut=1000): The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit.  
# 
# - **tol**[ float or None, optional (default=1e-3)]: The stopping criterion. If it is not None, the iterations will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs.  
# 
# - **shuffle**[bool, optional]: Whether or not the training data should be shuffled after each epoch. Defaults to True.  
# 
# - **epsilon**[float]: Epsilon in the epsilon-insensitive loss functions; only if loss is ‘huber’, ‘epsilon_insensitive’, or ‘squared_epsilon_insensitive’.
#   - For ‘huber’, determines the threshold at which it becomes less important to get the prediction exactly right. For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.  
#   
# - **random_state**[int, RandomState instance or None, optional (default=None)]: The seed of the pseudo random number generator to use when shuffling the data.
#   -  If None, the random number generator is the RandomState instance used by np.random.  
# 
# - **learning_rate**[string, optional]: learning rate schedule
#   - *constant*: eta = eta0
#   - *optimal*: $ eta = \frac{1.0}{alpha * ( t + t0)}$, t0 is chose by  a heuristic proposed by Leon Bottou
#   - *invscaling*[default]: $eta = \frac{eta0}{pow(t, 'power_t')}
#   - *adaptive*: eta = eta0, as long as the training keeps decreasing.
#     -  Each time n_iter_no_change consecutive epochs fail to decrease the training loss by tol or fail to increase validation score by tol if early_stopping is True, the current learning rate is divided by 5.  
#     
# - **eta0**[double]: The initial learning rate for the ‘constant’, ‘invscaling’ or ‘adaptive’ schedules. The default value is 0.01.  
# 
# - **power_t**[double]: The exponent for inverse scaling learning rate [default 0.5].  
# 
# - **early_stopping**[bool, default=False]: Whether to use early stopping to terminate training when validation score is not improving.
#   - If set to True, it will automatically set aside a fraction of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs.  
# 
# - **validation_fraction**[float, default=0.1]: The proportion of training data to set aside as validation set for early stopping.
#   - Must be between 0 and 1. Only used if early_stopping is True.  
#   
# - **n_iter_no_change**[int, default=5]: Number of iterations with no improvement to wait before early stopping.  
# 
# - **warm_start**[bool, optional]: When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
#   - Repeatedly calling fit or partial_fit when warm_start is True can result in a different solution than when calling fit a single time because of the way the data is shuffled. If a dynamic learning rate is used, the learning rate is adapted depending on the number of samples already seen. Calling fit resets this counter, while partial_fit will result in increasing the existing counter.  
#   
# - **average**[bool or int, optional]: When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute.
#   -  If set to an int greater than 1, averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples.  

# Attributes
# ---

# - **coef_** [array, shape (n_features,)]:  Weights assigned to the features.
# 
# - **intercept_**[array, shape (1,)]: The intercept term.
# 
# - **average_coef_**[array, shape (n_features,)]: Averaged weights assigned to the features.
# 
# - **average_intercept_ **[array, shape (1,)]: The averaged intercept term.
# 
# - **n_iter_**[int]: The actual number of iterations to reach the stopping criterion.

# 
# Regularization
# ---
# 

# 
# - regression의 결과 중 상관계수가 높은 것들에 집중하는 penalty를 주는 방법
# - 

# Setup
# ---

# In[ ]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
"""
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
"""

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# Get the data
# ============

# In[ ]:


"""
from google.colab import files
uploaded=files.upload()

for fn in uploaded.keys():
  print('User uploaded file"{name}" with length{length} bytes'.format(
      name = fn, length=len(uploaded[fn])
  ))"""


# In[ ]:


"""from google.colab import drive

drive.mount('/content/gdrive')"""


# In[ ]:


import pandas as pd

df = pd.read_csv("3MA_data.csv")
df.head()


# In[ ]:


df.info()


# 2 Variable (1k_RE & 1k_IM) data
# ---

# In[ ]:


df = df.loc[:,["yield stress", "elongation", "1k_RE", "1k_IM"]]
df = df.drop(df.index[190:209])
df.info()


# In[ ]:


df.describe()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins=50, figsize=(20,15))


# In[ ]:


# to make this notebook's output identical at every run
np.random.seed(42)


# In[ ]:


# train, test data split
from sklearn.model_selection import train_test_split

X = df.loc[:,["1k_RE","1k_IM"]]
ys = df.loc[:,"yield stress"]
elong = df.loc[:,"elongation"]

X_train, X_test, ys_train, ys_test = train_test_split(X, ys, test_size=0.2, random_state=42)
X_train, X_test, el_train, el_test = train_test_split(X, elong, test_size=0.2, random_state=42)
len(X_train)


# In[ ]:


len(X_test)


# In[ ]:


#X_train.to_csv("X_train.csv", mode='w')


# Discover and visualize the data to gain insights
# ===

# In[ ]:


# Copy the dataset in order not to harm train set
df_copy = df.copy()
df_copy.info()


# In[ ]:


X_train.info()


# In[ ]:


df_copy.plot(kind="scatter", x="1k_RE", y="yield stress")


# In[ ]:


df_copy.plot(kind="scatter", x="1k_IM", y="yield stress")


# In[ ]:


# Analysis of Standard correlation coefficient
corr_matrix = df_copy.corr()
corr_matrix


# In[ ]:


# Check what affects the most for the yield stress
corr_matrix["yield stress"].sort_values(ascending=False)

# 1kHz_voltage > 1k_RE > 1k_IM


# In[ ]:


corr_matrix["elongation"].sort_values(ascending=False)


# In[ ]:





# In[ ]:


# Feature Scaling => Standardization
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

data_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])
imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_train_std = data_pipeline.fit_transform(X_train)

ys_train = np.array(ys_train)
ys_train =ys_train.reshape(-1, 1)
ys_train = imputer.fit_transform(ys_train)
ys_train_std = data_pipeline.fit_transform(ys_train)

el_train = np.array(el_train)
el_train =el_train.reshape(-1, 1)
el_train = imputer.fit_transform(el_train)
el_train_std = data_pipeline.fit_transform(el_train)

X_test = imputer.fit_transform(X_test)
X_test_std =data_pipeline.fit_transform(X_test)

ys_test = np.array(ys_test)
ys_test = ys_test.reshape(-1, 1)
ys_test = imputer.fit_transform(ys_test)
ys_test_std = data_pipeline.fit_transform(ys_test)

el_test = np.array(el_test)
el_test = el_test.reshape(-1, 1)
el_test = imputer.fit_transform(el_test)
el_test_std = data_pipeline.fit_transform(el_test)


#print(X_train_std)

ys_test_std


# In[ ]:


#np.savetxt("ys_train.csv", ys_train, delimiter=",")


# In[ ]:


X_train_std.shape


# In[ ]:


ys_train


# Select and train a SGD regression model
# ===

# Yield Stress

# In[ ]:


# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "BestHyperParameter/SGD_Regression/yield_stress_2V"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


# In[ ]:


# Normal Stochastic Gradient Descent

from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
sgd_reg = SGDRegressor(random_state=42)
sgd_reg.fit(X_train_std, ys_train_std)
ys_predictions_sgd = sgd_reg.predict(X_train_std) * np.std(ys_train) + np.mean(ys_train)
sgd_mse = mean_squared_error(ys_train, ys_predictions_sgd)
sgd_rmse = np.sqrt(sgd_mse)
sgd_rmse


# Hyper Parameters Tuning: GridSearch CV function
# ---

# class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None, n_jobs=None, iid=’warn’, refit=True, cv=’warn’, verbose=0, pre_dispatch=‘2*n_jobs’, error_score=’raise-deprecating’, return_train_score=False)[source]

# GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
# 
# The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# ### parameters
# 
# - **estimator**[estimator object.]: This is assumed to implement the scikit-learn estimator interface. Either estimator needs to provide a score function, or scoring must be passed.
# 
# - **param_grid**[dict or list of dictionaries]: This enables searching over any sequence of parameter settings.
#   - Dictionary with parameters names (string) as keys and lists of parameter settings to try as values, or a list of such dictionaries, in which case the grids spanned by each dictionary in the list are explored.
#   
# - **scoring**[string, callable, list/tuple, dict or None, default: None]: A single string or a callable to evaluate the predictions on the test set.
#   - NOTE that when using custom scorers, each scorer should return a single value. Metric functions returning a list/array of values can be wrapped into multiple scorers that return one value each.
#   - If None, the estimator’s score method is used.
# 
# - **n_jobs**[int or None, optional (default=None)]:specify how many concurrent processes/threads should be used for parallelized routines.
#   - None means 1 unless in a joblib.parallel_backend context. 
#   - -1 means using all processors.
#   - Whether parallel processing is helpful at improving runtime depends on many factors, and it’s usually a good idea to experiment rather than assuming that increasing the number of jobs is always a good thing. It can be highly detrimental to performance to run multiple copies of some estimators or functions in parallel.
#   
# - **pre_dispatch**[int, or string, optional]: Controls the number of jobs that get dispatched during parallel execution
#   - Reducing this number can be useful to avoid an explosion of memory consumption when more jobs get dispatched than CPUs can process.
#     - None, in which case all the jobs are immediately created and spawned. Use this for lightweight and fast-running jobs, to avoid delays due to on-demand spawning of the jobs
#     - An int, giving the exact number of total jobs that are spawned
#     - A string, giving an expression as a function of n_jobs, as in ‘2*n_jobs’
#     
# - **iid**[boolean, default=’warn’]: If True, return the average score across folds, weighted by the number of samples in each test set.
#   - In this case, the data is assumed to be identically distributed across the folds, and the loss minimized is the total loss per sample, and not the mean loss across the folds. 
#   - If False, return the average score across folds. 
#   - Changed in version 0.20: Parameter iid will change from True to False by default in version 0.22, and will be removed in 0.24.
#   
# - **cv**[int, cross-validation generator or an iterable, optional]: Determines the cross-validation splitting strategy.
#   - possible inputs of cv are:
#     - None, to use the default 3-fold cross validation,
#     - integer, to specify the number of folds in a (Stratified)KFold,
#     - CV splitter,
#     - An iterable yielding (train, test) splits as arrays of indices.
#     - For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
#     
# - **refit**[ boolean, string, or callable, default=True]: Refit an estimator using the best found parameters on the whole dataset.
#   - For multiple metric evaluation, this needs to be a string denoting the scorer that would be used to find the best parameters for refitting the estimator at the end.
#   - The refitted estimator is made available at the best_estimator_ attribute and permits using predict directly on this GridSearchCV instance.
#   - Also for multiple metric evaluation, the attributes best_index_, best_score_ and best_params_ will only be available if refit is set and all of them will be determined w.r.t this specific scorer. best_score_ is not returned if refit is callable.
#   
# - **verbose**[int]: Controls the verbosity: the higher, the more messages.
# 
# - **error_score**[‘raise’ or numeric]: Value to assign to the score if an error occurs in estimator fitting.
#   - If set to ‘raise’, the error is raised. 
#   - If a numeric value is given, FitFailedWarning is raised. 
#   - Default is ‘raise’ but from version 0.22 it will change to np.nan.
#   
# - **return_train_score**[boolean, default=False]: If False, the cv_results_ attribute will not include training scores. 
#   - Computing training scores is used to get insights on how different parameter settings impact the overfitting/underfitting trade-off. 
#   - However computing the scores on the training set can be computationally expensive and is not strictly required to select the parameters that yield the best generalization performance.
# 

# ### Attributes
# 
# - **cv_results_**[dict of numpy (masked) ndarrays]: A dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
# 
# <div>
# {
# 'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
#                              mask = [False False False False]...)
#                              
# 'param_gamma': masked_array(data = [-- -- 0.1 0.2],
#                             mask = [ True  True False False]...),
#                             
# 'param_degree': masked_array(data = [2.0 3.0 -- --],
#                              mask = [False False  True  True]...),
#                              
# 'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
# 
# 'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
# 
# 'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
# 
# 'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
# 
# 'rank_test_score'    : [2, 4, 3, 1],
# 
# 'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
# 
# 'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
# 
# 'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
# 
# 'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
# 
# 'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
# 
# 'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
# 
# 'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
# 
# 'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
# 
# 'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
# 
# }
# </div>
# 
#   - For multi-metric evaluation, the scores for all the scorers are available in the cv_results_ dict at the keys ending with that scorer’s name ('_<scorer_name>') instead of '_score' shown above.
#   
# - **best_estimator**[estimator or dict]: Estimator that was chosen by the search,
#   -  i.e. estimator which gave highest score (or smallest loss if specified) on the left out data. Not available if refit=False.
#   
# - **best_score_**[float]:Mean cross-validated score of the best_estimator
# 
# - **best_params_**[dict]: Parameter setting that gave the best results on the hold out data.
# 
# - **best_index_**[int]: The index (of the cv_results_ arrays) which corresponds to the best candidate parameter setting.
#   - The dict at search.cv_results_['params'][search.best_index_] gives the parameter setting for the best model, that gives the highest mean score (search.best_score_).
#   
# - **scorer_**[function or a dict]: Scorer function used on the held out data to choose the best parameters for the model.
#   - For multi-metric evaluation, this attribute holds the validated scoring dict which maps the scorer key to the scorer callable.
# 
# - **n_splits_**[int]: The number of cross-validation splits (folds/iterations).
# 
# - **refit_time_**[float]: Seconds used for refitting the best model on the whole dataset.
#   - This is present only if refit is not False.
# 

# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    
    {'penalty': [ 'l2', 'l1', 'elasticnet'], 'alpha': [0.001, 0.005, 0.01,  0.05,  0.1],
     'l1_ratio': [0.2,0.4,,0.6,0.8], 'learning_rate': ['constant', 'optimal','invscaling','adaptive'],
    'eta0': [0.001, 0.01, 0.1],  'early_stopping':[True]}
  ]

sgd_reg = SGDRegressor(random_state=42)

grid_search = GridSearchCV(sgd_reg, param_grid, cv=30,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(X_train_std, ys_train_std)


# In[ ]:


grid_search.best_estimator_


# In[ ]:


grid_search.best_params_


# In[ ]:


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[ ]:


ys_sgd_final_model = grid_search.best_estimator_


# In[ ]:


ys_sgd_train_predictions = ys_sgd_final_model.predict(X_train_std) * np.std(ys_train) + np.mean(ys_train)
ys_sgd_train_mse = mean_squared_error(ys_train, ys_sgd_train_predictions)
ys_sgd_train_rmse = np.sqrt(ys_sgd_train_mse)
ys_sgd_train_rmse


# In[ ]:


ys_sgd_test_predictions = ys_sgd_final_model.predict(X_test_std)  * np.std(ys_test) + np.mean(ys_test)
ys_sgd_test_mse = mean_squared_error(ys_test,ys_sgd_test_predictions)
ys_sgd_test_rmse = np.sqrt(ys_sgd_test_mse)
ys_sgd_test_rmse


# ## Central Limit Theorem

# In[ ]:


print(cvres['mean_train_score'].shape)
print('Combinations')


# In[ ]:


mean_train_score = cvres['mean_train_score']
print(mean_train_score)


# In[ ]:


mean_test_score = cvres['mean_test_score']
print(mean_test_score)


# In[ ]:


std_train_score = cvres['std_train_score']
print(std_test_score)


# In[ ]:


std_test_score = cvres['std_test_score']
print(std_test_score)


# In[ ]:


#with open('/content/gdrive/My Drive/validation/cvscores/SGD/yield_stress/SGD_2V.txt', 'w') as f:
with open('/BestHyperParameter/SGD_Regression/yield_stress_2V/SGD_2V.txt', 'w') as f: 
  for key in cvres.keys():
    f.write("\n")
    f.write(key)
    f.write(": ")
    f.write(np.str(cvres[key]))
    f.write("\n")
    


get_ipython().system('cat /content/gdrive/My Drive/validation/cvscores/SGD/yield_stress/SGD_2V.txt')


# ## Train data plotting
# 

# In[ ]:


# Difference of train data
xx = np.linspace(0,len(X_train),len(X_train))
plt.figure
plt.grid()
ys_train_sort = np.sort(ys_train, axis=None)
ys_sgd_train_predictions_sort= np.sort(ys_sgd_train_predictions, axis=None)
plt.plot(xx,ys_train_sort,"b-", xx, ys_sgd_train_predictions_sort,"r--")
plt.title("Train data vs Train Prediction(SGD) Param Fitted")
plt.xlabel("Order of data")
plt.ylabel("Yield Stress")
save_fig("TrainPrediction_with_2V_SGD_ParamFitted")


# In[ ]:


# Difference of train data
difference = ys_train_sort - ys_sgd_train_predictions_sort

xx = np.linspace(0,len(difference),len(difference))
y = np.zeros((len(difference),1))
#y.reshape(1,len(difference))
plt.figure
plt.grid()
plt.plot(xx,y,"b-", xx, difference,"r--")
plt.title("Difference between Train data vs Prediction(SGD) Param Fitted")
#plt.ylim(-30, 30)
plt.xlabel("Order of data")
plt.ylabel("Yield Stress")

save_fig("Difference_between_Train_data_vs_Prediction_2V_SGD_ParamFitted")


# ## Test data plotting

# In[ ]:


# Difference of test data
xx = np.linspace(0,len(X_test),len(X_test))
plt.figure
plt.grid()
ys_test_sort = np.sort(ys_test, axis=None)
ys_sgd_test_predictions_sort= np.sort(ys_sgd_test_predictions, axis=None)
plt.plot(xx,ys_test_sort,"b-", xx, ys_sgd_test_predictions_sort,"r--")
plt.title("Test data vs Test Prediction(SGD) Param Fitted")
plt.xlabel("Order of data")
plt.ylabel("Yield Stress")
save_fig("TestPrediction_with_2V_SGD_ParamFitted")


# In[ ]:


# Difference of test data
difference = ys_test_sort - ys_sgd_test_predictions_sort

xx = np.linspace(0,len(difference),len(difference))
y = np.zeros((len(difference),1))
#y.reshape(1,len(difference))
plt.figure
plt.grid()
plt.plot(xx,y,"b-", xx, difference,"r--")
plt.title("Difference between Test data vs Prediction(SGD) Param Fitted")
#plt.ylim(-30, 30)
plt.xlabel("Order of data")
plt.ylabel("Yield Stress")

save_fig("Difference_between_Test_data_vs_Prediction_2V_SGD_ParamFitted")


# ## Extracting files

# In[ ]:


from google.colab import files
# Upload local files to Colab VM
#uploaded = files.upload()
# Download Colab VM fiels to local
files.download('images/BestHyperParameter/SGD_Regression/yield_stress_2V/TrainPrediction_with_2V_SGD_ParamFitted.png')


# In[ ]:


files.download('images/BestHyperParameter/SGD_Regression/yield_stress_2V/TestPrediction_with_2V_SGD_ParamFitted.png')


# In[ ]:


files.download('images/BestHyperParameter/SGD_Regression/yield_stress_2V/Difference_between_Train_data_vs_Prediction_2V_SGD_ParamFitted.png')


# In[ ]:


files.download('images/BestHyperParameter/SGD_Regression/yield_stress_2V/Difference_between_Test_data_vs_Prediction_2V_SGD_ParamFitted.png')


# In[ ]:


files.download('/content/gdrive/My Drive/validation/cvscores/SGD/SGD_2V.txt')


# In[ ]:





# #### elongation

# In[ ]:




