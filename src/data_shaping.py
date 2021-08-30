# QDT COMPETITION 2021 TEAM 3
# -----------------------------------------------------------------------------#-------------------- 
# THIS FILE IS FOR LOADING AND SHAPING DATA.
#
# CODED BY R. YAMAMOTO, T. AGO, A. KOBAYASHI
# LAST UPDATED ON 2021/8/31
# -----------------------------------------------------------------------------#--------------------

from typing import List, Tuple
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
import pandas as pd
import numpy as np
from contextlib import contextmanager
# from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import time
import gc

# データのPATH
PATH_APPLICATION = '../input/application.csv'
PATH_BUREAU = '../input/bureau.csv'
PATH_BUREAU_BALANCE = '../input/bureau_balance.csv'
PATH_PREVIOUS_APPLICATION = '../input/previous_application.csv'
PATH_POS_CASH_BALANCE = '../input/POS_CASH_balance.csv'
PATH_INSTALLMENTS_PAYMENTS = '../input/installments_payments.csv'
PATH_CREDIT_CARD_BALANCE = '../input/credit_card_balance.csv'
PATH_TRAIN_KEY = '../input/train_key.csv'
PATH_TEST_KEY = '../input/test_key.csv'

# 時間の表示
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-Hot エンコード
def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True) -> Tuple:
    original_columns = df.columns.to_list()
    categorical_columns = [col for col in original_columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [col for col in df.columns if col not in original_columns]
    return df, new_columns

# application.csv の前処理
def application(nan_as_category: bool = False) -> pd.DataFrame:
    df = pd.read_csv(PATH_APPLICATION)
    # 性別不詳（'XNA'）に該当する（4）行を除外
    df = df[df['CODE_GENDER'] != 'XNA']
    # バイナリエンコード
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    del uniques
    gc.collect()
    # One-Hot エンコード
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    del cat_cols
    gc.collect()
    # 'DAYS_EMPLOEYED'の誤植（？）365243をnanに変更
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # 簡単な特徴量追加
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df

# bureau.csv と bureau_balance.csv の前処理
def bureau_and_balance(nan_as_category: bool = True) -> pd.DataFrame:
    bureau = pd.read_csv(PATH_BUREAU)
    bb = pd.read_csv(PATH_BUREAU_BALANCE)
    # One-Hot エンコーディング
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(nan_as_category: bool = True) -> pd.DataFrame:
    prev = pd.read_csv('../input/previous_application.csv')
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= nan_as_category)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(nan_as_category: bool = True) -> pd.DataFrame:
    pos = pd.read_csv('../input/POS_CASH_balance.csv')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=nan_as_category)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(nan_as_category: bool = True) -> pd.DataFrame:
    ins = pd.read_csv('../input/installments_payments.csv')
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=nan_as_category)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(nan_as_category: bool = True) -> pd.DataFrame:
    cc = pd.read_csv('../input/credit_card_balance.csv')
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=nan_as_category)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

class preprocess():
    def __init__(self, target_table_names: List = None, is_test: bool = False):
        """
        dfs: application.csv以外で必要なテーブルデータのリスト, "bureau" or "prev" 
        or "pos_cash_balance" or "installment_payments" or "credit_card_balance"

        is_test: 本番（テスト）実施か否か
        """
        self.name_preprocess = {"bureau": bureau_and_balance,\
                                "prev": previous_applications,\
                                "pos_cash_balance": pos_cash,\
                                "installments_payments": installments_payments,\
                                "credit_card_balance": credit_card_balance}
        self.target_table_names = target_table_names
        if self.target_table_names:
            self.target_table_preprocesses = [self.name_preprocess[name] for name in target_table_names]
        self.is_test = is_test
    
    def get_train_test_data(self) -> Tuple:
        with timer("Data Shaping"):
            df = application()
        if self.target_table_names:
            for process in self.target_table_preprocesses:
                with timer("Data Shaping"):
                    df_add = process()
                    df = df.join(df_add, how='left', on='SK_ID_CURR')
                    del df_add
                    gc.collect()
        df_key_tr = pd.read_csv(PATH_TRAIN_KEY)
        train = df.merge(df_key_tr, how="inner", on="SK_ID_CURR")
        del df_key_tr
        gc.collect()
        if self.is_test:
            df_key_ts = pd.read_csv(PATH_TEST_KEY)
            test = df.merge(df_key_ts, how="inner", on="SK_ID_CURR")
            del df_key_ts
            gc.collect()
        else:
            train, test = train_test_split(train, stratify = train["TARGET"], test_size = 0.2)
        return train, test