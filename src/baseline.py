####################################################################################################
# QDT COMPETITION 2021 TEAM3
# Author: Ago, T., Kobayashi, A. and Yamamoto, R.
# Last edited: 2021/09/16
####################################################################################################
import time
from contextlib import contextmanager
import argparse # https://docs.python.org/ja/3/library/argparse.html
from typing import Tuple # https://docs.python.org/ja/3/library/typing.html

import numpy as np
from numpy.lib.npyio import load # https://numpy.org/doc/stable/reference/
import pandas as pd # https://pandas.pydata.org/docs/reference/index.html#api
from pandas.testing import assert_index_equal # https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_index_equal.html
from scipy.sparse.construct import random # https://docs.scipy.org/doc/scipy/reference/sparse.html
from scipy.stats import randint # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html
from sklearn.base import BaseEstimator, TransformerMixin # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base
from sklearn.compose import ColumnTransformer # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
from sklearn.impute import SimpleImputer # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, confusion_matrix, accuracy_score # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
from sklearn.pipeline import Pipeline # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
from sklearn.preprocessing import FunctionTransformer, LabelEncoder # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

import matplotlib.pyplot as plt

# 評価用
# try:
#     from evaluate import calculate_ar_from_pred_table
#     IS_EVALUATE = True
# except:
#     IS_EVALUATE = False


# パス（ローカルから）
# PATH_APPLICATION = "../input/application.csv"
# TRAIN_KEY = "../input/train_key.csv"
# TEST_KEY = "../input/train_test.csv"

# パス
PATH_APPLICATION = r"C:\Users\ymmtr\Dropbox\qdtComp\input\application.csv"
TRAIN_KEY = r"C:\Users\ymmtr\Dropbox\qdtComp\input\train_key.csv"
TEST_KEY = r"C:\Users\ymmtr\Dropbox\qdtComp\input\test_key.csv"

# パス
# PATH_APPLICATION = "s3://input/application.csv"
# TRAIN_KEY = "s3://input/train_key.csv"
# TEST_KEY = "s3://input/train_test.csv"

# 時間の表示
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# データ読み込み
def load_dataset(debug:bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nrows = 5000 if debug else None
    df_application = pd.read_csv(PATH_APPLICATION)
    df_tr_key = pd.read_csv(TRAIN_KEY)
    df_ts_key = pd.read_csv(TEST_KEY)
    train, test = pd.merge(df_application, df_tr_key, on="SK_ID_CURR"), pd.merge(df_application, df_ts_key, on="SK_ID_CURR")

    return modify_application(train), modify_application(test)

# application.csv の前処理
def modify_application(app:pd.DataFrame) -> pd.DataFrame:
    # 性別不詳（'XNA'）に該当する（4）行を除外
    df = app[app['CODE_GENDER'] != 'XNA'].copy()
    # 'DAYS_EMPLOEYED'の誤植（？）365243をnanに変更
    df.loc[:, 'DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    return df

def split_dataset(df:pd.DataFrame, col_target:str, random_state:int=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_tr, df_ts = train_test_split(df, stratify=df[col_target],random_state=random_state)
    return df_tr, df_ts

# ラベルエンコーダー
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None: 
        return

    # データフレーム型に変更
    @staticmethod
    def _to_dataframe(x) -> pd.DataFrame:
        return pd.DataFrame(x)

    # エンコーディング操作を決める
    def fit(self, X, y=None) -> None:
        df = self._to_dataframe(X)
        self.columns_ = df.columns
        self.les_ = dict()
        for c in df.columns:
            self.les_[c] = LabelEncoder() # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
            self.les_[c].fit(df[c]) # 例えば [1, 2, 2, 6] -> [0, 1, 1, 2] となる操作を決める
        return

    # self.fitしたあとで，そのエンコーディング操作を実行する
    def transform(self, X) -> pd.DataFrame:
        df = self._to_dataframe(X)
        assert_index_equal(df.columns, self.columns_) # df.columnsとself.columns_があっているか確認
        for c in df.columns:
            try:
                df[c] = self.les_[c].transform(df[c]) # エンコーディングする
            except ValueError:
                unknowns = set(df[c].unique()) - set(self.les_[c].classes_) # 入力データとエンコーディング前データ（self.les_[c].classes_）の差分をとる
                unknown_to_mode = {k: df[c].mode()[0] for k in unknowns} # unknownに属する値をdf[c]の最頻値に対応付ける
                df[c] = self.les_[c].transform(df[c].replace(unknown_to_mode)) # エンコーディングする
        return df
    
    # エンコーディング（fit + transform）
    def fit_transform(self, X, y=None) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)

# パイプライン
def get_pipeline(df_tr:pd.DataFrame, col_idx:str, col_target:str, is_random_search:bool=False, random_state:int=42) -> Pipeline:
    """
    col_idx : str
              申込者IDのカラム名（今回は"SK_ID_CURR"に対応）
    col_target : str
                 目的変数のカラム名（今回は"TARGET"）
    is_random_search : bool
                       ハイパーパラメタチューニング（ランダム）を行うかどうか
    random_state : int
                 ランダムフォレストに使う乱数のシード
    """
    # カテゴリ変数のカラム名リスト
    col_categorical = [col for col in df_tr.columns if df_tr[col].dtype == 'object' and col not in [col_idx, col_target]]
    # 数値変数のカラム名リスト
    col_numerical = [e for e in df_tr.columns if e not in col_categorical+[col_idx, col_target]]
    
    # 欠損値値補完に使う値
    fill_value_na = "#####"
    fill_value_numerical = df_tr[col_numerical].max().max() # （カラムに無関係に）数値データの最大値

    # カテゴリ変数の変換（欠損値処理＋エンコーディング）
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=fill_value_na)),
        ("label_encoder", MultiColumnLabelEncoder())
    ])

    # 数値変数の変換（欠損値処理）
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=fill_value_numerical))
    ])

    # 特徴量変換（カテゴリ変数と数値変数のまとめ）
    feature_transformer = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, col_categorical),
            ("numerical", numerical_transformer, col_numerical)
    ])

    # パイプライン
    ppl = Pipeline(steps=[
        ("transformer", feature_transformer),
        ("classifier", RandomForestClassifier(random_state=random_state))
    ])

    # ハイパラチューニングしないのであればそのままわたす
    if not is_random_search:
        return ppl
    else:
        parametes = {
            "classifier__max_depth": randint(1, 100),
            "classifier__max_features": ["sqrt", "log2", None],
            "classifier_min_samples_leaf": randint(2, 300)
        }
        return RandomizedSearchCV(ppl, parametes, n_iter=10, cv=5, random_state=random_state, n_jobs=-1, verbose=1)

class OurRandomForestClassifier():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--debug', action='store_true')
        self.parser.add_argument('--random_search', action='store_true')

        self.args = self.parser.parse_args(args=[]) #jupyter-notebookでエラーが起きたので永山さんのベースラインから変更
        self.df, self.df_eval = load_dataset(self.args.debug)

        self.col_idx = "SK_ID_CURR"
        self.col_target = "TARGET"

        self.df_tr, self.df_ts = split_dataset(self.df, self.col_target)

        self.pipeline = get_pipeline(self.df_tr, self.col_idx, self.col_target, is_random_search=self.args.random_search)

        # 学習
        self.pipeline.fit(self.df_tr.drop([self.col_target, self.col_idx], axis=1), self.df_tr[self.col_target])

        # デフォルト確率
        self.pred_train = self.pipeline.predict_proba(
            self.df_tr.drop([self.col_target, self.col_idx], axis=1))[:, 1]
        self.pred_test = self.pipeline.predict_proba(
            self.df_ts.drop([self.col_target, self.col_idx], axis=1))[:, 1]

    # AUC
    def get_auc_score(self, is_print:bool = True) -> Tuple[float, float]:
        auc_train = roc_auc_score(self.df_tr[self.col_target], self.pred_train)
        auc_test = roc_auc_score(self.df_ts[self.col_target], self.pred_test)
        if is_print:
            print("Train AUC: {}".format(auc_train))
            print("Test AUC: {}".format(auc_test))
        return auc_train, auc_test

    # AR
    def get_ar_score(self, is_print:bool = True) -> Tuple[float, float]:
        ar_train = 2 * roc_auc_score(self.df_tr[self.col_target], self.pred_train) - 1
        ar_test = 2 * roc_auc_score(self.df_ts[self.col_target], self.pred_test) - 1
        if is_print:
            print("Train AR {}".format(ar_train))
            print("Test AR: {}".format(ar_test))
        return ar_train, ar_test

    # ROC curve
    def plot_roc_curve_for_test(self, save_path:str=None) -> None:
        """
        save_path : str
                    保存フォルダ名
        """
        fpr_train, tpr_train, thresholds_train = roc_curve(self.df_tr[self.col_target], self.pred_train)
        fpr_test, tpr_test, thresholds_test = roc_curve(self.df_ts[self.col_target], self.pred_test)
        # お絵かき
        fig = plt.figure(figsize=(24, 12))
        ax_train, ax_test = fig.add_subplot(121), fig.add_subplot(122)

        ax_train.plot(fpr_train, tpr_train, marker="o")
        ax_train.set_xlabel("fpr")
        ax_train.set_ylabel("tpr")
        ax_train.set_title("ROC curve (train)")

        ax_test.plot(fpr_test, tpr_test, marker="o")
        ax_test.set_xlabel("fpr")
        ax_test.set_ylabel("tpr")
        ax_test.set_title("ROC curve (test)")
        plt.show()

        if save_path:
            fig.savefig(save_path)

    # accuracyを高める為に閾値を最適化
    def threshold_optimizer_on_accuracy_train(self, num:int=101) -> Tuple[float, float]:
        """
        num : int
              [0,1]を何分割するか
        """
        threshold_giving_max = 0
        max_accuracy = 0
        
        # thresholdによる閾値判別機
        def judge_default(x, threshold:float) -> int:
            if x > threshold:
                return 1
            else:
                return 0
        is_default = np.vectorize(judge_default)
        
        for threshold in np.linspace(0, 1, num=num):
            accuracy = accuracy_score(self.df_tr[self.col_target], is_default(self.pred_train, threshold))
            if max_accuracy < accuracy:
                max_accuracy = accuracy
                threshold_giving_max = threshold
        
        return max_accuracy, threshold_giving_max


        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--random_search', action='store_true')

    args = parser.parse_args()
    df, df_eval = load_dataset(args.debug)

    col_idx = "SK_ID_CURR"
    col_target = "TARGET"

    df_tr, df_ts = split_dataset(df, col_target)

    pipeline = get_pipeline(df_tr, col_idx, col_target, is_random_search=args.random_search)

    # 学習
    pipeline.fit(df_tr.drop([col_target, col_idx], axis=1), df_tr[col_target])

    # デフォルト率
    pred_train = pipeline.predict_proba(
        df_tr.drop([col_target, col_idx], axis=1))[:, 1]
    pred_test = pipeline.predict_proba(
        df_ts.drop([col_target, col_idx], axis=1))[:, 1]

    # AUC, AR
    print("Train AUC:")
    print(2 * roc_auc_score(df_tr[col_target], pred_train) - 1)

    print("Test AR:")
    print(2 * roc_auc_score(df_ts[col_target], pred_test) - 1)

    # コンペ運営用
    # if IS_EVALUATE:
    #     pred_pr_test =  pipeline.predict_proba(
    #         df_tr.drop([col_target, col_idx], axis=1))[:, 1]
    #     df_pred_pr_test = pd.DataFrame(
    #         {col_idx: df_eval[col_idx], "pred": pred_pr_test})
    #     ar_pr_test = calculate_ar_from_pred_table(df_pred_pr_test)
    #     print("Private Test AR")
    #     print(ar_pr_test)

if __name__ == "__main__":
    main()