#!/usr/bin/env python
# coding: utf-8

# 
# Title: Gradient Boosting Regressor for Property Prediction
# =======
# - Created: 2019.10.08
# - Updated: 2019.10.11
# - Author: Kyung Min, Lee
# 
# Learned from 
# - "Chapter 2 of Hands-on Machine Learning Book"
# - Sckit-Learn documents
#   -https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
# - https://subinium.github.io/introduction-to-ensemble-2-boosting/
# - greedy algorithm:https://janghw.tistory.com/entry/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-Greedy-Algorithm-%ED%83%90%EC%9A%95-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
# - https://3months.tistory.com/368
# - https://4four.us/article/2017/05/gradient-boosting-simply
# - https://wikidocs.net/19037
# - visualization: http://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html
# - https://soobarkbar.tistory.com/41
# - "Stochastic Gradient Boosting" paper of Jerome Friedman
# - https://3months.tistory.com/368
# 

# 

# ## GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage a regression tree is fit on the negative gradient of the given loss function.
# > class sklearn.ensemble.GradientBoostingRegressor(loss=’ls’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)[source]
# 
# - AdaBoost와 같은 순서로 진행합니다. 단 모델의 가중치(D)를 계산하는 방식에서 Gradient Descent를 이용하여 파라미터를 구합니다.
# 
# - AdaBoost에서는 모델 가중치를 고려한 선형합으로 최종 predictions을 구합니다. 그렇다면 이 모델 가중치 선형 합 연산을 하나의 식으로 본다면 어떨까요? 모델에 따른 최적의 가중치를 Gradient decent로 구해서 보다 최적화된 결과를 얻고자 하는 것이 GBM의 특징입니다.
# 
# 

# ### Gradient Boosting 
# - - -
# - Gradient Descent는 손실함수를 파라미터로 미분해서 기울기를 구하고, 값이 작아지는 방향으로 파라미터를 움직이다 보면 손실함수가 최소화되는 지점에 도딜한다. 이 과정을 함수공간에서 진행한다. 그래서 손실함수를 파라미터가 아니라 현재까지 학습된 모델 함수로 미분한다.
# 
# $$ f_{i+1} = f_i + \rho \frac{\partial J}{\partial f_i} $$
# 
# - 파라미터 공간에서는 계산된 기울기를 따라서 학습률(Learning Rate, ρ)에 맞춰 θ를 업데이트하면 된다.
# 
# - Gradient Boosting은 이 미분값을 다음 모델(Weak Learner)의 타겟으로 넘긴다. Squared Error를 쓰는 경우를 예로 들면, 현재 모델의 잔차를 타겟으로 놓고 새로운 모델 피팅을 한다. 기존 모델은 이 새로운 모델을 흡수해서 Bias를 줄인다. 그리고 다시 잔차를 구하고 모델을 피팅해서 더하기를 반복한다. 매우 단순하고 직관적인 방법인데, 이걸 손실함수를 L2로 설정한 Gradient Boosting으로 설명할 수 있다.
# 
# -  Gradient Boosting에서는 Gradient가 현재까지 학습된 모델의 약점을 드러내는 역할을 하고, 다른 모델이 그걸 중점적으로 보완해서 성능을 Boosting한다. 위에서는 L2 손실함수를 썼지만 미분만 가능하다면 다른 손실함수도 얼마든지 쓸 수 있다는 것이 장점이다. 
# 
# - 부스팅 알고리즘의 특성상 계속 약점(오분류/잔차)을 보완하려고 하기 때문에 잘못된 레이블링이나 아웃라이어에 필요 이상으로 민감할 수 있다. 이런 문제에 강인한 L1 Loss나 Huber Loss를 쓰고자 한다면 그냥 손실함수만 교체하면 된다. 손실함수의 특성은 Gradient를 통해 자연스럽게 학습에 반영된다.
# 

# 
# ![example](https://wikidocs.net/images/page/19037/tree_O9zyAlk.png)
# ex) 앙상블 모델의 예측값 ui는 각 트리의 결과를 가중 합산해서 계산할 수 있다. (여기서 Tj(xi)는 트리 j가 xi를 입력으로 받아 출력한 결과이다.)
# $$ u_i = \sum_{j=1}^{M} \beta_j T_j(x_i) $$
# Loss 함수의 경우 관측값과 예측값의 오차가 최소화되도록 오차 제곱의 합 형태인 
# $L=(y_i,u_i)=(y_i−u_i)2$로 정의할 수 있다.
# $$ min_\beta \sum_{i=1}^{n} L(y_i,  \sum_{j=1}^{M} \beta_j T_j(x_i)) $$
# 
# - 일반적으로 앙상블 모델에서 트리 구성을 할 때는 고정 depth를 갖는 작은 트리를 아주 여러개 만든다. 왜냐하면 트리를 작게 하면 메모리도 적게 사용하고 예측도 빠르게 할 수 으며 트리의 개수가 많아질 수록 앙상블의 성능은 좋아지게 되기 때문이다. 일반적으로 트리의 depth는 5이하로 고정한다.
# 
# - 이 문제를 풀려면 최적화 문제를 좀 더 쉬운 문제로 바꿔야 한다. 원래 최적화 문제는 Loss 함수를 최소화하는 M개의 가중치 βj를 찾는 문제이다. 이 문제를 예측값 u에 대한 함수 f(u)를 최소화 문제 $min_u f(u)$로 생각해 보자. 함수 f(u)가 Loss 함수 L(y,u)라고 하면 Loss 함수를 최소화 하는 u를 찾는 것이 쉽게 재정의된 문제라고 할 수 있다. (여기서 n은 데이터 개수이다.)
# 
# - Gradient boosting는 $min_u f(u)$로 재정의된 최소화 문제를 gradient descent를 이용해서 풀는 기법을 말한다.
# 

# ## Algorithm

# - Gradient boosting 알고리즘은 $min_u L(y,u)$의 최적해 $u^∗$를 찾기 위해 다음과 같은 방식으로 gradient descent를 수행한다.
# 
# 1. 초기 값은 임의의 트리의 결과 값으로 $u^(0)=T_0$와 같이 설정한다. 그리고, 다음의 2~4 단계를 반복한다.
# 
# 2. n 개의 데이터의 가장 최근의 예측값인 $u^(k−1)$에 대한 음수 gradient를 계산한다.
# $$ d_i = -[\frac{\partial L(y_i, u_i)}{\partial u_i} |_{u_i=u_i^(k-1)}, i = 1,...,n $$
# 
# 3. n 개 데이터에 대한 gradient di와 트리의 결과 $T(x_i)$가 가장 비슷한 트리 $T_k$를 찾는다.
# $$ min_{trees T} \sum_{i=1}^{n} (d_i - T(x_i))^2 $$
# 
# 4. Step size $a_k$를 계산하고 위에서 찾은 $T_k$를 이용하여 예측값을 업데이트한다.
# $$ u^(k) = u^(k-1) + \alpha_k T_k $$
# 
# - 이 알고리즘은 gradient descent로 최적해 $u^∗$를 찾기 위해 u에 대한 gradient d를 구하고, g에 가장 가까운 $T_k$를 찾아서 업데이트 식에 gradient 대신 $T_k$를 대입해서 다음 위치를 구한다.
# 
# - 이렇게 해서 구한 최종 예측값 $u^∗$는 앞에서 정의했던 트리 결과의 가중 합산과 동일해짐을 알 수 있다. (즉, 재귀식 형태의 업데이트 식 $u^{k}=u^{k−1}+α_k T_k$을 k=0까지 풀어보면 $u^∗ = \sum_{k=1}^{n} \alpha_k T_k$가 되어 트리 결과의 가중 합산 형태로 만들 수 있다. )
# 

# HyperParameters
# ---
# 

# - **loss**[‘ls’, ‘lad’, ‘huber’, ‘quantile’}, optional (default=’ls’)]: loss function to be optimized.
#   - *‘ls’* refers to least squares regression.
#   - *‘lad’* (least absolute deviation) is a highly robust loss function solely based on order information of the input variables. 
#   - *‘huber’* is a combination of the two. 
#   - *‘quantile’* allows quantile regression (use alpha to specify the quantile).
# 
# - **learning_rate**[float, optional (default=0.1)]: 
#   - learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators.
# 
# - **n_estimators**[int (default=100)]: The number of boosting stages to perform. 
#   - Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
# 
# - **subsample**[float, optional (default=1.0)]: The fraction of samples to be used for fitting the individual base learners. 
#   - If smaller than 1.0 this results in Stochastic Gradient Boosting. subsample interacts with the parameter n_estimators. Choosing subsample < 1.0 leads to a reduction of variance and an increase in bias.
# 
# - **criterion**[string, optional (default=”friedman_mse”)]: The function to measure the quality of a split. 
#   - Supported criteria are “friedman_mse” for the mean squared error with improvement score by Friedman, “mse” for mean squared error, and “mae” for the mean absolute error. The default value of “friedman_mse” is generally the best as it can provide a better approximation in some cases.
# 
# -  **min_samples_split**[int, float, optional (default=2)]: The minimum number of samples required to split an internal node
#   - If int, then consider min_samples_split as the minimum number.
#   - If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
# 
# 
# - **min_samples_leaf**[int, float, optional (default=1)]: The minimum number of samples required to be at a leaf node. 
#   - A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
#   - If int, then consider min_samples_leaf as the minimum number.
#   - If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
# 
# 
# - **min_weight_fraction_leaf**[float, optional (default=0.)]: The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
# 
# - **max_depth**[integer, optional (default=3), 4~8 best??]: maximum depth of the individual regression estimators. 
#   - The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables.
# 
# - **min_impurity_decrease**[float, optional (default=0.)]: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
# 
# The weighted impurity decrease equation is the following:
# 
# $$ N_t / N * (impurity - N_t_R / N_t * right_impurity - N_t_L / N_t * left_impurity) $$
# where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.
# 
# 
# N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
# 
# 
# 
# - **min_impurity_split**[float, (default=1e-7)]: Threshold for early stopping in tree growth. 
#   - A node will split if its impurity is above the threshold, otherwise it is a leaf.
# 
# Deprecated since version 0.19: Use min_impurity_decrease instead.
# 
# - **init**[estimator or ‘zero’, optional (default=None)]: An estimator object that is used to compute the initial predictions. 
#   - init has to provide fit and predict. If ‘zero’, the initial raw predictions are set to zero. By default a DummyEstimator is used, predicting either the average target value (for loss=’ls’), or a quantile for the other losses.
# 
# - **random_state**[int, RandomState instance or None, optional (default=None)]:
# 
#   - If int, random_state is the seed used by the random number generator;
#   - If RandomState instance, random_state is the random number generator;
#   - If None, the random number generator is the RandomState instance used by np.random.
# 
# - **max_features**[int, float, string or None, optional (default=None)]: The number of features to consider when looking for the best split:
# 
#   - If int, then consider max_features features at each split.
#   - If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
#   - If “auto”, then max_features=n_features.
#   - If “sqrt”, then max_features=sqrt(n_features).
#   - If “log2”, then max_features=log2(n_features).
#   - If None, then max_features=n_features.
#   
# Choosing max_features < n_features leads to a reduction of variance and an increase in bias.
# 
# Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
# 
# - **alpha**[float (default=0.9)]: The alpha-quantile of the huber loss function and the quantile loss function. Only if loss='huber' or loss='quantile'.
# 
# - **verbose**[int, default: 0]: Enable verbose output. If 1 then it prints progress and performance once in a while (the more trees the lower the frequency). If greater than 1 then it prints progress and performance for every tree.
# 
# - **max_leaf_nodes**[int or None, optional (default=None)]: Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
# 
# - **warm_star**[bool, default: False]: When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution. See the Glossary.
# 
# - **presort**[bool or ‘auto’, optional (default=’auto’)]: Whether to presort the data to speed up the finding of best splits in fitting. 
#   - Auto mode by default will use presorting on dense data and default to normal sorting on sparse data. 
#   - Setting presort to true on sparse data will raise an error.
# 
# New in version 0.17: optional parameter presort.
# 
# - **validation_fraction**[float, optional, default 0.1]: The proportion of training data to set aside as validation set for early stopping.
#   - Must be between 0 and 1. Only used if n_iter_no_change is set to an integer.
# 
# - **n_iter_no_change**[int, default None]: n_iter_no_change is used to decide if early stopping will be used to terminate training when validation score is not improving. 
#   - By default it is set to None to disable early stopping. If set to a number, it will set aside validation_fraction size of the training data as validation and terminate training when validation score is not improving in all of the previous n_iter_no_change numbers of iterations.
# 
# 
# - **tol**[float, optional, default 1e-4]: Tolerance for the early stopping. 
#   - When the loss is not improving by at least tol for n_iter_no_change iterations (if set to a number), the training stops.
# 
# New in version 0.20.

# Attributes
# ---

# - **feature_importances_**[array, shape (n_features,)]: Return the feature importances (the higher, the more important the feature).
# 
# - **oob_improvement_**[array, shape (n_estimators,)]: The improvement in loss (= deviance) on the out-of-bag samples relative to the previous iteration. oob_improvement_[0] is the improvement in loss of the first stage over the init estimator.
# 
# - **train_score_**[array, shape (n_estimators,)]: The i-th score train_score_[i] is the deviance (= loss) of the model at iteration i on the in-bag sample. If subsample == 1 this is the deviance on the training data.
# 
# - **loss_**[LossFunction]: The concrete LossFunction object.
# 
# - **init_**[estimator]: The estimator that provides the initial predictions. Set via the init argument or loss.init_estimator.
# 
# - **estimators_**[array of DecisionTreeRegressor, shape (n_estimators, 1)]: The collection of fitted sub-estimators.
# 
# 

# Method
# ---

# - **apply(self, X)**	Apply trees in the ensemble to X, return leaf indices.
# - **fit(self, X, y[, sample_weight, monitor])**	Fit the gradient boosting model.
# - **get_params(self[, deep])**	Get parameters for this estimator.
# - **predict(self, X)**	Predict regression target for X.
# - **score(self, X, y[, sample_weight])**	Returns the coefficient of determination R^2 of the prediction.
# - **set_params(self, \*\*params)**	Set the parameters of this estimator.
# - **staged_predict(self, X)**	Predict regression target at each stage for X.

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


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")


# In[ ]:


sklearn.__version__


# Get the data
# ============

# In[ ]:


"""
from google.colab import files
uploaded=files.upload()

for fn in uploaded.keys():
  print('User uploaded file"{name}" with length{length} bytes'.format(
      name = fn, length=len(uploaded[fn])
  ))
  """


# In[ ]:


"""
from google.colab import drive

drive.mount('/content/gdrive')
"""


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


print(X_train)


# ## Feature Scaling

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


#np.savetxt("ys_train.csv", ys_train, delimiter=",")
#np.savetxt("x_train_std.csv", X_train_std, delimiter=",")


# In[ ]:


X_train_std.shape


# In[ ]:


ys_train.shape


# In[ ]:


X_train_std


# In[ ]:


np.random.seed(42)


# Select and train a gradient boosting regression model
# ===

# In[ ]:


# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "BestHyperParameter/GB_Regression/yield stress_2V"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


# In[ ]:


# Normal Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

gb_reg = GradientBoostingRegressor(random_state=42)
gb_reg.fit(X_train_std, ys_train_std)
ys_predictions_gb= gb_reg.predict(X_train_std) * np.std(ys_train) + np.mean(ys_train)
gb_mse = mean_squared_error(ys_train, ys_predictions_gb)
gb_rmse = np.sqrt(gb_mse)
gb_rmse


# Hyper Parameters Tuning: GridSearch CV function
# ---

# GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
# 
# The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.

# In[ ]:


from sklearn.model_selection import GridSearchCV


param_grid = [
    
 {'loss': ['ls', 'lad', 'huber', 'quantile'], 'learning_rate': [0.001, 0.005, 0.01], 'n_estimators': [100,150,200,300],
  'min_samples_leaf': [1,2,3,4],
  'max_depth': [3,5,7], 'random_state': [42], 'max_features': [0.7,0.85,1],
  'n_iter_no_change': [5,10,20,30,40]
    }
  ]

gb_reg = GradientBoostingRegressor(random_state=42)

grid_search = GridSearchCV(gb_reg, param_grid, cv=30,
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


ys_gb_final_model = grid_search.best_estimator_


# In[ ]:


X_test_std = np.array(X_test_std)
#X_test_std= X_test_std.reshape(-1, 1)
#X_test_std


# In[ ]:


ys_gb_train_predictions = ys_gb_final_model.predict(X_train_std) * np.std(ys_train) + np.mean(ys_train)
ys_gb_train_mse = mean_squared_error(ys_train, ys_gb_train_predictions)
ys_gb_train_rmse = np.sqrt(ys_gb_train_mse)
ys_gb_train_rmse


# In[ ]:


ys_gb_test_predictions = ys_gb_final_model.predict(X_test_std) * np.std(ys_test) + np.mean(ys_test)
ys_gb_test_mse = mean_squared_error(ys_test,ys_gb_test_predictions)
ys_gb_test_rmse = np.sqrt(ys_gb_test_mse)
ys_gb_test_rmse


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


#with open('/content/gdrive/My Drive/validation/cvscores/Gradient_boost/yield_stress/GB_2V.txt', 'w') as f:
with open('BestHyperParameter/GB_Regression/yield stress_2V/GB_2V.txt', 'w') as f:
  for key in cvres.keys():
    f.write("\n")
    f.write(key)
    f.write(": ")
    f.write(np.str(cvres[key]))
    f.write("\n")


#!cat /content/gdrive/My Drive/validation/cvscores/Adaboost/yield_stress/ADB_2V.txt


# Train data plotting
# ---

# In[ ]:


# Difference of train data
xx = np.linspace(0,len(X_train),len(X_train))
plt.figure
plt.grid()
ys_train_sort = np.sort(ys_train, axis=None)
ys_gb_train_predictions_sort= np.sort(ys_gb_train_predictions, axis=None)
plt.plot(xx,ys_train_sort,"b-", xx, ys_gb_train_predictions_sort,"r--")
plt.title("Train data vs Train Prediction(GB) Param Fitted")
plt.xlabel("Order of data")
plt.ylabel("Yield Stress")
save_fig("TrainPrediction_with_2V_GB_ParamFitted")


# In[ ]:


# Difference of train data
difference = ys_train_sort - ys_gb_train_predictions_sort

xx = np.linspace(0,len(difference),len(difference))
y = np.zeros((len(difference),1))
#y.reshape(1,len(difference))
plt.figure
plt.grid()
plt.plot(xx,y,"b-", xx, difference,"r--")
plt.title("Difference between Train data vs Prediction(GB) Param Fitted")
#plt.ylim(-30, 30)
plt.xlabel("Order of data")
plt.ylabel("Yield Stress")

save_fig("Difference_between_Train_data_vs_Prediction_2V_GB_ParamFitted")


# Test data plotting
# ---

# In[ ]:


# Difference of train data
xx = np.linspace(0,len(X_test),len(X_test))
plt.figure
plt.grid()
ys_test_sort = np.sort(ys_test, axis=None)
ys_gb_test_predictions_sort= np.sort(ys_gb_test_predictions, axis=None)
plt.plot(xx,ys_test_sort,"b-", xx, ys_gb_test_predictions_sort,"r--")
plt.title("Test data vs Test Prediction(GB) Param Fitted")
plt.xlabel("Order of data")
plt.ylabel("Yield Stress")
save_fig("TestPrediction_with_2V_GB_ParamFitted")


# In[ ]:


# Difference of train data
difference = ys_test_sort - ys_gb_test_predictions_sort

xx = np.linspace(0,len(difference),len(difference))
y = np.zeros((len(difference),1))
#y.reshape(1,len(difference))
plt.figure
plt.grid()
plt.plot(xx,y,"b-", xx, difference,"r--")
plt.title("Difference between Test data vs Prediction(GB) Param Fitted")
#plt.ylim(-30, 30)
plt.xlabel("Order of data")
plt.ylabel("Yield Stress")

save_fig("Difference_between_Test_data_vs_Prediction_2V_GB_ParamFitted")


# ## Extracting files

# In[ ]:


#!ls images/BestHyperParameter/GB_Regression/yield_stress_2V/


# In[ ]:


"""
from google.colab import files
# Upload local files to Colab VM
#uploaded = files.upload()
# Download Colab VM fiels to local
files.download('images/BestHyperParameter/GB_Regression/yield_stress_2V/TrainPrediction_with_2V_GB_ParamFitted.png')
"""


# In[ ]:


#files.download('images/BestHyperParameter/GB_Regression/yield_stress_2V/TestPrediction_with_2V_GB_ParamFitted.png')


# In[ ]:


#files.download('images/BestHyperParameter/GB_Regression/yield_stress_2V/Difference_between_Train_data_vs_Prediction_2V_GB_ParamFitted.png')


# In[ ]:


#files.download('images/BestHyperParameter/GB_Regression/yield_stress_2V/Difference_between_Test_data_vs_Prediction_2V_GB_ParamFitted.png')


# In[ ]:


#files.download('/content/gdrive/My Drive/validation/cvscores/Gradient_Boost/yield_stress/GB_2V.txt')


# In[ ]:





# In[ ]:





# #### elongation

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




