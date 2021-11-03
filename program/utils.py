import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
# validation
from sklearn.metrics import mean_squared_error

INPUT = "../input"
SUBMIT = "../submit"
TRAIN = os.path.join(INPUT, 'train.csv')
TEST = os.path.join(INPUT, 'test.csv')
ALL = os.path.join(INPUT, 'all.csv')
FEATURE = "../feature"
TARGET = 'loan_status'

"""入出力関係"""

def read_csv(path):
    df = pd.read_csv(path)
    return df


def save_feature(df):
    """
    生成した特徴量をcsvに保存する関数
    """
    dt_now = datetime.datetime.now()
    now =dt_now.strftime("%m%d%H%M")
    path = os.path.join(FEATURE, now + ".csv")
    print(path)
    df.to_csv(path, index=False)
    print("save completed")


def read_new_csv():
    """
    最新のcsvを読む関数
    基本的には最新の特徴量を入手する用途
    """
    csv_path = glob(FEATURE + "/*.csv")
    csv_path = sorted(csv_path, reverse=True)
    print(f"{csv_path[0]} was read")
    df = read_csv(csv_path[0])
    return df


def to_submit_csv(df):
    # dfの整形
    # 要修正
    df_test = read_csv(TEST)
    df["IDppp"] =df_test["ID"]
    df = df.set_axis(['y', "ID"], axis=1)
    df = df.reindex(columns=['ID', 'y'])
    dt_now = datetime.datetime.now()
    now =dt_now.strftime("%m%d%H%M")
    path = os.path.join(SUBMIT, now + "_submit.csv")
    print(path)
    df.to_csv(path, index = False)
    print("save completed")


def df_split(df):
    """
    is_trainの有無で訓練か検証かを分ける
    """
    # trainを整形
    df_train = df[df["is_train"]]
    df_train = df_train.drop(["is_train"], axis=1)
    #testを整形
    df_test = df[df["is_train"] == False]
    df_test = df_test.drop(["is_train"] + [TARGET], axis=1)
    return (df_train, df_test)


"""モデルの訓練と評価"""

def val(pred, test):
    mse = mean_squared_error(pred, test)
    rmse = np.sqrt(mse)
    return rmse

def validate(model, X_train, y_train, X_test, y_test):
    # train model using train data
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = val(pred, y_test)
    return score

"""DataFrameの整形"""
def train_test_merge(train_df, test_df, is_save = False):
    """
    trainとtestを後から区別できるように結合する
    """
    train_df["is_train"] = True
    test_df["is_train"] = False
    df = pd.concat([train_df, test_df])
    if is_save:
        df.to_csv(ALL, index = False)
    return df