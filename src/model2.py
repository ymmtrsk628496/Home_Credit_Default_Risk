#################################################################################
# QDT COMPETITION 2021 TEAM3 (Fairness in Credit Scoring)
# Author: Ago, T., Kobayashi, A. and Yamamoto, R.
# BASELINE
# Last edited: 2021/09/17
#################################################################################

# ライブラリー
import argparse # https://docs.python.org/ja/3/library/argparse.html
from typing import Tuple # https://docs.python.org/ja/3/library/typing.html
import numpy as np # https://numpy.org/doc/stable/reference/
import pandas as pd # https://pandas.pydata.org/docs/reference/index.html#api
from pandas.testing import assert_index_equal # https://pandas.pydata.org/docs/reference/api/pandas.tes ting.assert_index_equal.html
from scipy.sparse.construct import random # https://docs.scipy.org/doc/scipy/reference/sparse.html
from scipy.stats import randint # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html
from sklearn.base import BaseEstimator, TransformerMixin # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.base
from sklearn.compose import ColumnTransformer # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.compose
from sklearn.ensemble import RandomForestClassifier # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
from sklearn.impute import SimpleImputer # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute
from sklearn.metrics import roc_auc_score # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
from sklearn.pipeline import Pipeline # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
from sklearn.preprocessing import LabelEncoder # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
from sklearn.linear_model import LogisticRegression # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
from lightgbm import LGBMClassifier # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
import matplotlib.pyplot as plt # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html

# 親パス
PATH_PARENT  = "../input/" # ローカルから
# PATH_PARENT = "s3://input/" # S3から
# データパス
PATH_APPLICATION =  PATH_PARENT + "application.csv"
TRAIN_KEY = PATH_PARENT + "train_key.csv"
TEST_KEY = PATH_PARENT + "test_key.csv"

def modify_application(app:pd.DataFrame):
    """
    applicationデータを加工

    Parameters
    ----------
    app: pd.DataFrame
        対象のapplicationデータ
    
    Returns
    -------
    modified_app: pd.DataFrame
        前処理済みのapplicationデータ
    """
    # 性別不詳（'XNA'）に該当する（4）行を除外
    modified_app = app[app['CODE_GENDER'] != 'XNA'].copy()
    # 'DAYS_EMPLOEYED'の誤植（？）365243をnanに変更
    modified_app.loc[:, 'DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    """
    applicationデータのその他の前処理はここに追加
    """
    return modified_app

"""
その他のデータの前処理はここに追加
"""

def load_dataset():
    """
    データの読み込み

    Parameters
    ----------
    None

    Returns
    -------
    train: pd.DataFrame
        訓練データ
    test: pd.DataFrame
        テストデータ
    """
    df_application = pd.read_csv(PATH_APPLICATION)
    df_tr_key = pd.read_csv(TRAIN_KEY)
    df_ts_key = pd.read_csv(TEST_KEY)
    train, test = pd.merge(df_application, df_tr_key, on="SK_ID_CURR"), pd.merge(df_application, df_ts_key, on="SK_ID_CURR")
    train, test = modify_application(train), modify_application(test)
    return train, test

def split_dataset(df:pd.DataFrame, col_target:str, random_state:int=42):
    """
    入力データを訓練データと検証データに分割する
    
    Parameters
    ----------
    df: pd.DataFrame
        入力訓練データ
    col_targets: str
        目的変数の列名
    random_state: int
        乱数のシード
    
    Returns
    -------
    df_tr: pd.DataFrame
        訓練データ
    df_ts: pd.DataFrame
        検証データ
    """
    df_tr, df_ts = train_test_split(df, stratify=df[col_target],random_state=random_state)
    return df_tr, df_ts

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """
    エンコーディングの処理を保持する
    """
    def __init__(self):
        """
        Parameters
        ----------
        None
        """
        return

    @staticmethod
    def _to_dataframe(x):
        """
        配列をDataFrame型にする

        Parameters
        ----------
        x: ndarray (structured or homogeneous), Iterable, dict, or DataFrame
            入力データ
        Returns
        -------
        pd.DataFrame
        """
        return pd.DataFrame(x)
    
    def fit(self, X, y=None):
        """
        エンコーディング操作を定義する

        Parameters
        ----------
        X: ndarray (structured or homogeneous), Iterable, dict, or DataFrame
            入力データ
        y: None
            固定

        Returns
        =======
        None
        """
        df = self._to_dataframe(X)
        self.columns_ = df.columns
        self.les_ = dict()
        for col in df.columns:
            self.les_[col] = LabelEncoder() # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
            self.les_[col].fit(df[col]) # 例えば [1, 2, 2, 6] -> [0, 1, 1, 2] となる操作を決める
        return

    def transform(self, X):
        """
        self.fitしたあとで，そのエンコーディング操作を実行する

        Parameters
        ----------
        X: ndarray (structured or homogeneous), Iterable, dict, or DataFrame
            入力データ

        Returns
        -------
        df: pd.DataFrame
            エンコーディング後データ    
        """  
        df = self._to_dataframe(X)
        assert_index_equal(df.columns, self.columns_) # df.columnsとself.columns_があっているか確認
        for col in df.columns:
            try:
                df[col] = self.les_[col].transform(df[col]) # エンコーディングする
            except ValueError:
                unknowns = set(df[col].unique()) - set(self.les_[col].classes_) # 入力データとエンコーディング前データ（self.les_[c].classes_）の差分をとる
                unknown_to_mode = {k: df[col].mode()[0] for k in unknowns} # unknownsに属する値をdf[c]の最頻値に対応付ける
                df[col] = self.les_[col].transform(df[col].replace(unknown_to_mode)) # エンコーディングする
        return df
    
    def fit_transform(self, X, y=None):
        """
        エンコーディング処理を実行

        Parameters
        ----------
        X: ndarray (structured or homogeneous), Iterable, dict, or DataFrame
            入力データ
        y: None
            固定

        Returns
        -------
        pd.DataFrame
            エンコーディング後データ    
        """  
        self.fit(X)
        return self.transform(X)

"""
LogisticRegression用のエンコーダーを追加（OneHotEncoding, Scaling, 欠損処理）
回帰モデルのため，Scalingを行いたい
"""

def get_pipeline(df_tr:pd.DataFrame, col_idx:str, col_target:str, model:str, random_state:int=42):
    """
    パイプラインを作成

    Parameters
    ----------
    df_tr: pd.DataFrame
        訓練データ
    col_idx: str
        申込者IDのカラム名（今回は"SK_ID_CURR"に対応）
    col_target: str
        目的変数のカラム名（今回は"TARGET"）
    model: str
        モデル名（lgbm, rf）
    random_state: int
        ランダムフォレスト. LGBMに使う乱数のシード
    
    Returns
    -------
    Pipeline
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
    if model == "rf":
        ppl = Pipeline(steps=[
            ("transformer", feature_transformer),
            ("classifier", RandomForestClassifier(random_state=random_state))
        ])
    if model == "lgbm":
        ppl = Pipeline(steps=[
            ("transformer", feature_transformer),
            ("classifier", LGBMClassifier(random_state=random_state))
        ])
    return ppl


"""
LogisticRegression用のパイプラインを追加
"""

"""
fairnessスコアの関数たち
"""

"""
評価スコアの関数たち
"""
def get_DP(df:pd.DataFrame, sensitive_attribute:str, pred_train):
    """
    Demographic parityを計算

    Parameters
    ----------
    pred_train
    """
    pass

def get_EO(df:pd.DataFrame, sensitive_attribute:str, pred_train):
    """
    Demographic parityを計算

    Parameters
    ----------
    pred_train
    """
    pass

# Fairnessとのトレードオフを考えない単純な学習
class OurModel():
    """
    モデルのメイン部分を実行

    Attributes
    ----------
    model: str
        選択したモデル（lgbm, rf）
    df: pd.DataFrame
        読み込みデータ
    df_eval: pd.DataFrame
        検証データ
    col_idx: str
        ID列名
    col_target: str
        目的変数列名
    df_tr: pd.DataFrame
        訓練データ
    df_ts: pd.DataFrame
        テストデータ
    pipelines: dict
        モデル名と関数名の辞書
    pipeline: Pipeline
        選択したパイプライン
    best_params_: 
    """
    def __init__(self, model:str, is_search:bool=True, search_type="random", random_state=42):
        """
        Parameters
        ----------
        model: str
            使うモデルの名前（rf, lgbm）
        is_search: bool
            ハイパラ探索するか
        search_type: str
            探索方法（random, grid）
        random_state: int
            ランダム探索の乱数シード
        """
        # 使うモデル
        self.model = model

        # データ読み込み（訓練用，検証用）
        self.df, self.df_eval = load_dataset()

        # カラム名（id，target）
        self.col_idx = "SK_ID_CURR"
        self.col_target = "TARGET"

        # 訓練データ分割
        self.df_tr, self.df_ts = split_dataset(self.df, self.col_target)

        # pipeline取得
        pipeline = get_pipeline(self.df_tr, self.col_idx, self.col_target, model=model)
        if not is_search:
            # 学習
            pipeline.fit(self.df_tr.drop([self.col_target, self.col_idx], axis=1), self.df_tr[self.col_target])
            self.pipeline = pipeline
        else:
            # ランダムフォレストの場合
            if model == "rf":
                # parametersいじって探索してください
                parameters = {
                    "classifier__n_estimators": [10, 100, 1000],
                    "classifier__max_depth": [10, 20, 30],
                    "classifier__max_features": ["sqrt", "log2", None],
                    "classifier__min_samples_leaf": [10, 20, 30],
                    "classifier__min_samples_split": [10, 20, 30]
                }
                # ランダムサーチ
                if search_type == "random":
                    pipeline = RandomizedSearchCV(pipeline, parameters, n_iter=10, cv=5, random_state=random_state, scoring='roc_auc', n_jobs=-1, verbose=1)
                # グリッドサーチ
                if search_type == "grid":
                    param_grid = parameters
                    pipeline =  GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
            # LGBMの場合
            if model == "lgbm":
                # parametersいじって探索してください
                parameters = {
                    "classifier__n_estimators": [10000],
                    # "classifier__learning_rate": [0.1 * i for i in range(1, 3)],
                    # "classifier__num_leaves": randint(30, 50),
                    # "classifier__reg_alpha": [0.01 * i for i in range(5)],
                    # "classifier__reg_lambda": [0.01 * i for i in range(5, 10)],
                    # "classifier__colsample_bytree": [0.9, 0.95, 1],
                    # "classifier__subsample": [0.9, 0.95, 1],
                    # "classifier__max_depth": randint(10, 15)
                    "classifier__learning_rate":[0.02],
                    "classifier__num_leaves":[34],
                    "classifier__colsample_bytree":[0.95],
                    "classifier__subsample":[0.9],
                    "classifier__max_depth":[10],
                    "classifier__reg_alpha":[0.04],
                    "classifier__reg_lambda":[0.07],
                    "classifier__min_split_gain":[0.02],
                    "classifier__min_child_weight":[39]
                }
                # ランダムサーチ
                if search_type == "random":
                    pipeline = RandomizedSearchCV(pipeline, parameters, n_iter=1, cv=5, random_state=random_state, scoring='roc_auc', n_jobs=-1, verbose=1)
                # グリッドサーチ
                if search_type == "grid":
                    param_grid = parameters
                    pipeline = GridSearchCV(pipeline, param_grid, cv=5, random_state=random_state, scoring='roc_auc', n_jobs=-1, verbose=1)
            # 学習
            pipeline.fit(self.df_tr.drop([self.col_target, self.col_idx], axis=1), self.df_tr[self.col_target])
            self.best_params_ = pipeline.best_params_
            self.cv_results_ = pipeline.cv_results_
            self.pipeline = pipeline

        # デフォルト確率
        self.pred_train = self.pipeline.predict_proba(
            self.df_tr.drop([self.col_target, self.col_idx], axis=1))[:, 1]
        self.pred_test = self.pipeline.predict_proba(
            self.df_ts.drop([self.col_target, self.col_idx], axis=1))[:, 1]

# Fairness制約付き学習（作りかけ．scikit-legoモジュールを使ってもいいが，オリジナルFairness指標にも対応させたい）
class FairnessConstrainedOurModel():
    def __init__(self):
        pass



# デバッグ用
def main(is_rf:bool=True, is_lgbm:bool=False, is_search:bool=True): 
    """
    Parameters
    ----------
    is_rf: bool
        rfを使うか
    is_lgbm: bool
        lgbmを使うか
    """
    if is_lgbm:
        model = "lgbm"
    if is_rf:
        model = "rf"
    # 学習        
    clf = OurModel(model=model, is_search=is_search)
    # AUC, AR
    print(f"Train AR (model):")
    print(2 * roc_auc_score(clf.df_tr[clf.col_target],clf.pred_train) - 1)
    print(f"Test AR (model):")
    print(2 * roc_auc_score(clf.df_ts[clf.col_target], clf.pred_test) - 1)
    if is_search:
        print(clf.best_params_)
        pd.DataFrame(clf.cv_results_).to_csv("./result_3.csv", index=False)

if __name__ == "__main__":
    main()