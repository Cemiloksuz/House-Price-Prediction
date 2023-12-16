########################## HOUSE PRICE PREDICTION PROJECT  ###########################

# This project aims to create a house price prediction model
# with 79 explanatory variables for residential homes in Ames, Iowa.
# In the project, the data was prepared for the machine learning model through data analysis and variable engineering.
# Then different machine learning models were tested and the best one was selected.


########################## Import Library and Settings  ###########################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import os
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score

warnings.simplefilter("ignore", category = ConvergenceWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

path = "C:\\Users\\hseym\\PycharmProjects\\pythonProject1\\miuul_homework\\datasets\\home_price"
os.chdir(path)

########################## Loading  The Data  ###########################
train = pd.read_csv("/Users/cemiloksuz/PycharmProjects/EuroTechMiullDataScience/week_11/train.csv")
test = pd.read_csv("/Users/cemiloksuz/PycharmProjects/EuroTechMiullDataScience/week_11/test.csv")

train.head()
test.head()

house_price = pd.concat([train, test], ignore_index = True)
df = house_price.copy()
df.head()
df.tail()


########################## Summary of  The Data  ###########################
def check_df(dataframe):
    if isinstance(dataframe, pd.DataFrame):
        print("########## shape #########\n", dataframe.shape)
        print("########## types #########\n", dataframe.dtypes)
        print("########## head #########\n", dataframe.head())
        print("########## tail #########\n", dataframe.tail())
        print("########## NA #########\n", dataframe.isna().sum())
        print("########## describe #########\n", dataframe.describe().T)
        print("########## nunique #########\n", dataframe.nunique())


check_df(df)


def columns_info(dataframe):
    columns, dtypes, unique, nunique, nulls = [], [], [], [], []

    for cols in dataframe.columns:
        columns.append(cols)
        dtypes.append(dataframe[cols].dtype)
        unique.append(dataframe[cols].unique())
        nunique.append(dataframe[cols].nunique())
        nulls.append(dataframe[cols].isnull().sum())

    return pd.DataFrame({"Columns": columns,
                         "Data_Type": dtypes,
                         "Unique_Values": unique,
                         "Number_of_Unique": nunique,
                         "Missing_Values": nulls})


columns_info(df)


########################## Cleaning of Some Weirdness ###########################
df.drop(columns = ["Id"], inplace = True)

df["MSSubClass"] = df["MSSubClass"].astype("object")
# MSSubClass is a categorical variable representing the housing type. Therefore its type has been changed to object.

df["MoSold"] = df["MoSold"].astype("object")
df["YrSold"] = df["YrSold"].astype("object")
# Sales year and month are in categorical format. Therefore the type has been changed to object.

df.loc[df["YearBuilt"] > df["YearRemodAdd"], "YearBuilt"] = df["YearRemodAdd"]
# Construction year of houses whose construction year was greater than renovation year was assigned as renovation year

df.loc[(df["MasVnrType"] == "None") & (df["MasVnrArea"] > 0.0), ["MasVnrType"]] = "other"
df.loc[(df["MasVnrType"].isna()) & (df["MasVnrArea"] > 0.0), ["MasVnrArea"]] = 0.0
df.loc[df["MasVnrType"].isna(), "MasVnrType"] = "None"
# Structural arrangements were made regarding the type of wall covering.


df.loc[df["GarageYrBlt"] == 2207, "GarageYrBlt"] = 2007
df.loc[df["GarageYrBlt"].isnull(), "GarageYrBlt"] = 0.0
df["GarageYrBlt"] = df["GarageYrBlt"].astype("int64")
# An arrangement was made regarding the garage construction years.

df.loc[df["Functional"].isna(), "Functional"] = "Typ"
# Fixed as typical if home functionality is empty.
df.loc[df["SaleType"].isna(), "SaleType"] = "Oth"
# Fixed as other if sale type is empty.


########################## Grab to Columns ###########################
def grab_col_names(dataframe, cat_th = 13, car_th = 30):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return list(cat_cols), list(num_cols), list(cat_but_car)

# Numeric columns were assigned categorically if their unique values were less than 13.(month is critical value for this)
# Neighborhood is critical value for cardinal categoric columns.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# It was aimed to determine hierarchical columns for encoding operations.
hierarchic_cols = [col for col in cat_cols if col.__contains__("Qu") |
                   (col.__contains__("Cond") & ~col.__contains__("Condition"))]

df[hierarchic_cols].head()
hierarchic_cols.extend(["BsmtExposure", "BsmtFinType1", "BsmtFinType2", "GarageFinish",
                        "Fence", "BsmtFullBath", "BsmtHalfBath", "GarageCars", "HeatingQC",
                        "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "Fireplaces"])

non_hierarchic_cols = [col for col in cat_cols if col not in hierarchic_cols]
df[non_hierarchic_cols].head()


########################## Variables Analysis ###########################
na_no_cols = ["Alley", "GarageType", "FireplaceQu", "GarageQual", "GarageCond", "GarageFinish", "Fence",
              "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual", "BsmtCond"]
# Columns whose NA status is significant and NA values will be replaced with No.

for col in na_no_cols:
    df.loc[df[col].isna(), col] = "No"

# Class numbers and ratios of categorical columns were analyzed.
def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block = True)


for col in cat_cols:
    cat_summary(df, col, True)
#for col in cat_cols[:3]:
#    cat_summary(df, col, True)


# Descriptive statistical analysis of numerical columns was performed and histograms were drawn.
def num_summary(dataframe, numerical_col, plot = False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        sns.histplot(dataframe[numerical_col])
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)


for col in num_cols:
    num_summary(df, col, plot = True)
#for col in num_cols[:3]:
#    num_summary(df, col, plot = True)


########################## Target Analysis ###########################
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end = "\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)


########################## Outlier Analysis ###########################
def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


for col in num_cols:
    print(check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(check_outlier(df, col))

# Threshold values were determined for numerical values and outliers were replaced with threshold values.

########################## Missing Value Analysis ###########################
def missing_values_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ['n_miss', 'ratio'])
    print(missing_df, end = "\n")

    if na_name:
        return na_columns


na_cols = missing_values_table(df, True)

drop_cols = ["PoolQC", "MiscFeature"]
# Missing value ratios 99.66, 96.40
df.drop(columns = drop_cols, inplace = True)

cat_cols = [col for col in cat_cols if col not in drop_cols]
non_hierarchic_cols = [col for col in non_hierarchic_cols if col not in drop_cols]
na_cols = [col for col in na_cols if col not in drop_cols]

msno.matrix(df[na_cols])
plt.show()


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end = "\n\n\n")


missing_vs_target(df, "SalePrice", na_cols)
missing_vs_target(df, "SalePrice", ["LotFrontage"])

df_clean = df.copy()
na_cols = na_cols[:-1]   # Sale Price is  not included.

objects = [col for col in na_cols if df_clean[col].dtype == "object"]
for col in objects:
    df_clean.loc[df_clean[col].isna(), col] = df_clean[col].mode()[0]

no_objects = [col for col in na_cols if col not in objects]
for col in no_objects:
    df_clean.loc[df_clean[col].isna(), col] = df_clean[col].median()

# Missing values in object columns were filled with mode, and missing values in numeric columns were filled with median.

missing_values_table(df_clean, True)


########################## Ordinal Encode ###########################
""" 1 """
six_level_columns = ["ExterQual", "ExterCond", "KitchenQual", "BsmtQual", "BsmtCond",
                     "FireplaceQu", "GarageQual", "GarageCond", "HeatingQC"]
levels = {"No": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}

for i in six_level_columns:
    df_clean[i] = df_clean[i].replace(levels)

df_clean[six_level_columns].head()

""" 2 """
other_level_col = [col for col in hierarchic_cols if col not in six_level_columns and df_clean[col].dtype == "object"]

bsmt_exp_level = {'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}
bsmt_fin_level = {'No': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
garage_fin_level = {'No': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
fence_level = {"No": 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}

df_clean["BsmtExposure"] = df_clean["BsmtExposure"].replace(bsmt_exp_level)
df_clean["BsmtFinType1"] = df_clean["BsmtFinType1"].replace(bsmt_fin_level)
df_clean["BsmtFinType2"] = df_clean["BsmtFinType2"].replace(bsmt_fin_level)
df_clean["GarageFinish"] = df_clean["GarageFinish"].replace(garage_fin_level)
df_clean["Fence"] = df_clean["Fence"].replace(fence_level)

df_clean[other_level_col].head()

""" 3 """
float_level_col = [col for col in hierarchic_cols if col not in six_level_columns and
                   col not in other_level_col and
                   df_clean[col].dtype == "float"]
for i in float_level_col:
    df_clean[i] = df_clean[i].astype("int64")

df_clean[float_level_col].head()


########################## Rare Analysis ###########################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")


rare_analyser(df_clean, "SalePrice", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis = None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# Among the classes of categorical columns,
# those with a ratio lower than 0.01 were collected and assigned to the rare class.

df_clean = rare_encoder(df_clean, 0.01)
df_clean.head()
rare_analyser(df_clean, "SalePrice", cat_cols)

df_temp = df_clean.copy()


########################## Feature Engineering ###########################
# Total Floor Area
df_clean['NEW_TotalSF'] = df_clean['TotalBsmtSF'] + df_clean['1stFlrSF'] + \
                          df_clean['2ndFlrSF'] + df_clean['LowQualFinSF']

# Total Quality parameters
df_clean["NEW_TotalQual"] = df_clean[
    ["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
     "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual",
     "GarageCond", "Fence"]].sum(axis = 1)

# Total Finished Basement Area
df_clean["NEW_TotalBsmtFin"] = df_clean["BsmtFinSF1"] + df_clean["BsmtFinSF2"]

# Total Porch Area
df_clean["NEW_PorchArea"] = df_clean["OpenPorchSF"] + df_clean["EnclosedPorch"] + df_clean["ScreenPorch"] + \
                            df_clean["3SsnPorch"] + df_clean["WoodDeckSF"]

# Garage product Living Area
df_clean["NEW_Garage*GrLiv"] = (df_clean["GarageArea"] * df_clean["GrLivArea"])

# Lot Ratio
df_clean["NEW_LotRatio"] = df_clean["GrLivArea"] / df_clean["LotArea"]

df_clean["NEW_RatioArea"] = df_clean["NEW_TotalSF"] / df_clean["LotArea"]

df_clean["NEW_GarageLotRatio"] = df_clean["GarageArea"] / df_clean["LotArea"]

# MasVnrArea
df_clean["NEW_MasVnrRatio"] = df_clean["MasVnrArea"] / df_clean["NEW_TotalSF"]

# Dif Area
df_clean["NEW_EmptyArea"] = (df_clean["LotArea"] - df["1stFlrSF"] - df_clean["GarageArea"] -
                             df_clean["NEW_PorchArea"] - df_clean["PoolArea"])

# Overall Condition product Quality
df_clean["NEW_OverallGrade"] = df_clean["OverallQual"] * df_clean["OverallCond"]

# Difference between built and restoration
df_clean["NEW_Restoration"] = df_clean["YearRemodAdd"] - df_clean["YearBuilt"]

# House Age
df_clean["NEW_HouseAge"] = (df_clean["YrSold"]).astype(int) - df_clean["YearBuilt"].astype(int)

# Restorated House Age
df_clean["NEW_RestorationAge"] = df_clean["YrSold"].astype(int) - df_clean["YearRemodAdd"].astype(int)

# Garage Age
df_clean["NEW_GarageAge"] = df_clean["YrSold"].astype(int) - df_clean["GarageYrBlt"].astype(int)

# House has Pool
df_clean['NEW_HasPool'] = df_clean['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

# House has 2ndFloor
df_clean['NEW_Has2ndfloor'] = df_clean['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

# Is it new?
df_clean["NEW_NewHouse"] = np.where((df_clean["YearRemodAdd"] - df_clean["YearBuilt"] > 0), 1, 0)

# Total bath number
df_clean["NEW_TotalBath"] = df_clean["BsmtFullBath"] + df_clean["FullBath"] + df_clean["BsmtHalfBath"] * 0.5 + \
                            df_clean["HalfBath"] * 0.5
# Has Basement
df_clean["NEW_HasBsmt"] = np.where(df["BsmtQual"] == "No", 0, 1)


########################## Label Encode ###########################
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df_clean.columns if (df_clean[col].dtype not in ["int64", "float64", "int32"])
               and (df_clean[col].nunique() == 2)]

for col in binary_cols:
    label_encoder(df_clean, col)

df_clean[binary_cols].head()


def one_hot_encoder(dataframe, categorical_cols, drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe


ohe_cols = [col for col in non_hierarchic_cols if df_clean[col].nunique() > 2]
df_clean[ohe_cols].head()

new_df = one_hot_encoder(df_clean, ohe_cols)
new_df.head()
new_df.shape

new_df.info()
# All columns were transformed to numeric values.

########################## Standardization ###########################
standard_col = [col for col in new_df.columns if new_df[col].dtype not in ["int32", "int64", "uint8"]
                or new_df[col].max() > 10]

standard_col.remove("SalePrice")

new_df[standard_col].head()

ss = StandardScaler()
for col in standard_col:
    new_df[col] = ss.fit_transform(new_df[[col]])
new_df.head()

new_df["SalePrice"] = np.log(new_df["SalePrice"])
# Logarithmic transformation was applied to the sales price column.


########################## Test Train Split ###########################
train_df = new_df.loc[new_df["SalePrice"].notna()]
test_df = new_df.loc[new_df["SalePrice"].isna()].reset_index(drop = True)
train_df.shape
test_df.shape

y = train_df["SalePrice"]
X = train_df.drop("SalePrice", axis = 1)


########################## Model Selection ###########################
models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor(random_state = 17)),
          ('RF', RandomForestRegressor(random_state = 17)),
          ('GBM', GradientBoostingRegressor(random_state = 17)),
          ("XGBoost", XGBRegressor(objective = 'reg:squarederror')),
          ("LightGBM", LGBMRegressor(random_state = 17))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv = 5, scoring = "neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 3700326299.0262 (LR)
# RMSE: 0.1594 (KNN)
# RMSE: 0.1941 (CART)
# RMSE: 0.1362 (RF)
# RMSE: 0.1245 (GBM)
# RMSE: 0.1391 (XGBoost)
# RMSE: 0.129 (LightGBM)

########################### Model  ###########################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 17)

lgbm_model = LGBMRegressor(random_state = 17)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X_train, y_train, cv = 5, scoring = "neg_mean_squared_error")))
# 0.13466041743996787 train hatası

lgbm_model = LGBMRegressor(random_state = 17).fit(X_train, y_train)

y_pred = lgbm_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
# 0.13043056414678503 test hatası

y_test[0:5]
y_pred[0:5]
y.mean()
# 12.02392324335732
y.std()
# 0.39898549750300966


########################### Hyperparameter Optimization ###########################
lgbm_params = {"learning_rate": [0.01, 0.05, 0.1, 0.2, 0.25],
               "n_estimators": [500, 1000, 1500, 2000, 2500],
               "colsample_bytree": [ 0.5, 0.7, 1]
               }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv = 5,
                            n_jobs = -1,
                            verbose = True).fit(X, y)

print(lgbm_gs_best.best_params_)

best = {'colsample_bytree': 0.5, 'learning_rate': 0.05, 'n_estimators': 2500}

final_model = lgbm_model.set_params(**best).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv = 5, scoring = "neg_mean_squared_error")))
# 0.12195787727784227


########################### Feature Importance  ###########################
def plot_importance(model, features, num = len(X), save = False):
    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value", ascending = False)[0:20])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")


plot_importance(final_model, X)


########################### Generate Predictions  ###########################
predictions = final_model.predict(test_df.drop(["SalePrice"], axis=1))
# test_df.drop(["SalePrice"]   x_test olarak düşünülmeli
predictions = np.exp(predictions)

submission = pd.DataFrame({"Id": test["Id"], "SalePrice": predictions})
submission.to_csv("HousePricePredictions.csv", index=False)
submission
