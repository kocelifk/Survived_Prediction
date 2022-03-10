############################################
# Gerekli Kütüphane ve Fonksiyonlar
############################################

import pickle
import pandas as pd
from helpers.eda import *
from helpers.data_prep import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
     roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


############################################
# EDA ANALIZI
############################################

def load():
    df = pd.read_csv("datasets/titanic.csv")
    df.columns = [str(i).upper() for i in list(df.columns)]
    return df


df = load()
check_df(df)

# BAĞIMLI DEĞİŞKEN ANALİZİ
df["SURVIVED"].describe()
sns.distplot(df["SURVIVED"])
plt.show()

sns.boxplot(df["SURVIVED"])
plt.show()

# KATEGORİK VE NUMERİK DEĞİŞKENLERİN SEÇİLMESİ
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

# KATEGORİK DEĞİŞKEN ANALİZİ
rare_analyser(df, "SURVIVED", cat_cols)

# SAYISAL DEĞİŞKEN ANALİZİ
for col in num_cols:
    num_summary(df, col, plot=True)

# AYKIRI GÖZLEM ANALİZİ
for col in num_cols:
    print(col, check_outlier(df, col))

# Eksik Gözlemler kontrol ediliyor.
missing_values_table(df)

############################################
# FEATURE ENGINEERING
############################################

df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr.")]))
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
cat_cols, num_cols, cat_but_car = grab_col_names(df)

#: Outliers
for col in num_cols:
    replace_with_thresholds(df, col, q1=0.25, q3=0.75)

#: Missing Values
df.drop("CABIN", inplace=True, axis=1)
remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x,axis=0)

#: Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

#: Rare Encoding
df = rare_encoder(df, 0.01)

#: One-Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#: Standart Scaler
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

############################################
# Model Kurulumu
############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

lgr = LogisticRegression(random_state=12345)
lgr_model = lgr.fit(X_train, y_train)


def control_y_pred(report_test, report_train=None):
    if report_train is not None:
        df = pd.DataFrame(report_train).T
        print("############## Y TRAIN ##############")
        print(df)

    df = pd.DataFrame(report_test).T
    print("############## Y TEST ##############")
    print(df)


y_train_pred = lgr_model.predict(X_train)
report_train = classification_report(y_train, y_train_pred, output_dict=True)

y_pred = lgr_model.predict(X_test)
report_test = classification_report(y_test, y_pred, output_dict=True)

control_y_pred(report_test, report_train)



