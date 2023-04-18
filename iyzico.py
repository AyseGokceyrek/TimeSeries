

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

df = pd.read_csv('datasets/iyzico_data.csv', index_col=[0])


###############################################################
# STEP 1 : EDA
###############################################################


def df_summary(df):
    print("\n" + 20 * "*" + "SHAPE".center(20) + 20 * "*")
    print("\n")
    print(df.shape)
    print("\n" + 20 * "*" + "INDEX".center(20) + 20 * "*")
    print("\n")
    print(df.index)
    print("\n" + 20 * "*" + "COLUMNS".center(20) + 20 * "*")
    print("\n")
    print(df.columns)
    print("\n" + 20 * "*" + "DATAFRAME INFORMATIONS".center(20) + 20 * "*")
    print("\n")
    print(df.info())
    print("\n"+ 20 * "*" + "DATAFRAME INFORMATIONS".center(20) + 20 * "*")
    print("\n")
    print(df.describe().T)
    print("\n" + 20 * "*" + "MISSING VALUES".center(20) + 20 * "*")
    print(df.isnull().sum())


df_summary(df)

df["transaction_date"] = df["transaction_date"].apply(pd.to_datetime)

# Start and end date of the dataset
df["transaction_date"].min(), df["transaction_date"].max()

# Observe the total number of transactions in each merchant
df["merchant_id"].unique()
df.groupby("merchant_id").agg({"Total_Transaction": "sum"})

# Observe the total amount of payment in each category

df["Total_Paid"].dtypes
df.loc[df["merchant_id"]].agg("Total_Paid").sum()

df.groupby("merchant_id").agg({"Total_Paid": ["sum", "mean", "median"]})

# Observe the transaction count graphs of the categories in each year

for id in df.merchant_id.unique():
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1, title = str(id) + ' 2018-2019 Transaction Count')
    df[(df.merchant_id == id) & (df.transaction_date >= "2018-01-01") & (df.transaction_date < "2019-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.subplot(3,1,2, title = str(id) + ' 2019-2020 Transaction Count')
    df[(df.merchant_id == id) & (df.transaction_date >= "2019-01-01") & (df.transaction_date < "2020-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.show()

###############################################################
# STEP 2: Feature Engineering
###############################################################
# Date Features


def create_date_features(df):
    df['month'] = df.transaction_date.dt.month
    df['day_of_month'] = df.transaction_date.dt.day
    df['day_of_year'] = df.transaction_date.dt.dayofyear
    #df['week_of_year'] = df.transaction_date.dt.weekofyear
    df['week_of_year'] = df.transaction_date.dt.isocalendar().week
    df['week_of_year'] = df['week_of_year'].astype(np.int32).astype(np.int32)
    df['day_of_week'] = df.transaction_date.dt.dayofweek
    df['year'] = df.transaction_date.dt.year
    df['is_wknd'] = pd.to_datetime(df['transaction_date']).dt.weekday.isin([5, 6]).astype(int)  # haftasonu olma durumu alışverişi etkileyen bir unsurdur.
    df['is_month_start'] = df.transaction_date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.transaction_date.dt.is_month_end.astype(int)
    return df

create_date_features(df)

df_summary(df)

df.groupby(["merchant_id", "year", "month", "is_wknd"]).agg({"Total_Transaction": ["sum", "mean"]})
df.groupby(["merchant_id","year","month"]).agg({"Total_Paid": ["sum", "mean", "median"]})

# Lag/Shifted Features


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe)))


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["merchant_id"])['Total_Transaction'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe


df = lag_features(df, [91,92,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,
                       350,351,352,352,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,
                       538,539,540,541,542, 718, 719, 720, 721,722])  # 3-month period

# Rolling Mean Features


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby("merchant_id")['Total_Transaction']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720])

# Exponentially Weighted Mean Features


def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby("merchant_id")['Total_Transaction'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe


alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720]

df = ewm_features(df, alphas, lags)
df.tail()
df.shape

###############################################################
# STEP 3 : MODEL
###############################################################
# 1.One-hot encoding
df = pd.get_dummies(df, columns=['merchant_id', 'day_of_week', 'month'])
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)

df_summary(df)

# 2.Custom Cost Functions
# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# 3.Split the dataset into train and validation
import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# Train set until the 10th month of 2020.
train = df.loc[(df["transaction_date"] < "2020-10-01"), :]

# The last 3 months of 2020 validation set
val = df.loc[(df["transaction_date"] >= "2020-10-01"), :]

# choosing the independent variable
cols = [col for col in train.columns if col not in ['transaction_date', 'id', "Total_Transaction","Total_Paid", "year" ]]


Y_train = train['Total_Transaction']
X_train = train[cols]

Y_val = val['Total_Transaction']
X_val = val[cols]


Y_train.shape, X_train.shape, Y_val.shape, X_val.shape


########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'],
                  feval=lgbm_smape,
                  verbose_eval=100)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))



def plot_lgb_importances(model, plot=False, num=10):

    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))


plot_lgb_importances(model, num=30, plot=True)

lgb.plot_importance(model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()
