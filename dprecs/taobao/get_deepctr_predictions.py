#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import copy
import warnings

import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd

from deepctr.feature_column import SparseFeat,VarLenSparseFeat,DenseFeat,get_feature_names
from deepctr.models import DIN, FNN
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import utils
warnings.simplefilter('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=".", help='Where to look for and store data')
args = parser.parse_args()

# Load data
raw_sample_df = pd.read_csv(f'{args.data_path}/raw_sample.csv')
ad_feature_df = pd.read_csv(f'{args.data_path}/ad_feature.csv')
user_profile_df=pd.read_csv(f'{args.data_path}/user_profile.csv')

def mem_usage(pandas_obj):
  if isinstance(pandas_obj,pd.DataFrame):
    usage_b = pandas_obj.memory_usage(deep=True).sum()
  else: # we assume if not a df it's a series
    usage_b = pandas_obj.memory_usage(deep=True)
  usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
  return "{:03.2f} MB".format(usage_mb)


test_size_mb = raw_sample_df.memory_usage().sum() / 1024 / 1024
test_size_mb1 = ad_feature_df.memory_usage().sum() / 1024 / 1024
test_size_mb2 = user_profile_df.memory_usage().sum() / 1024 / 1024
print("raw_sample_df memory size: %.2f MB" % test_size_mb)
print("ad_feature_df memory size: %.2f MB" % test_size_mb1)
print("user_profile_df memory size: %.2f MB" % test_size_mb2)

# raw_sample_df memory usage optimization
raw_sample_df.info(memory_usage='deep')
optimized_gl = raw_sample_df.copy()
gl_int = raw_sample_df.select_dtypes(include=['int'])
converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
optimized_gl[converted_int.columns] = converted_int
gl_obj = raw_sample_df.select_dtypes(include=['object']).copy()
converted_obj = pd.DataFrame()
for col in gl_obj.columns:
    num_unique_values = len(gl_obj[col].unique())
    num_total_values = len(gl_obj[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:,col] = gl_obj[col].astype('object')
    else:
        converted_obj.loc[:,col] = gl_obj[col]

optimized_gl[converted_obj.columns] = converted_obj
print("Original Ad Feature dataframe:{0}".format(mem_usage(raw_sample_df)))
print("Memory Optimised Ad Feature dataframe:{0}".format(mem_usage(optimized_gl)))
raw_sample_df = optimized_gl.copy()
raw_sample_df_new = raw_sample_df.rename(columns = {"user": "userid"})

# ad_feature_df memory usage optimization
ad_feature_df.info(memory_usage='deep')
optimized_g2 = ad_feature_df.copy()
g2_int = ad_feature_df.select_dtypes(include=['int'])
converted_int = g2_int.apply(pd.to_numeric,downcast='unsigned')
optimized_g2[converted_int.columns] = converted_int
g2_float = ad_feature_df.select_dtypes(include=['float'])
converted_float = g2_float.apply(pd.to_numeric,downcast='float')
optimized_g2[converted_float.columns] = converted_float
print("Original Ad Feature dataframe:{0}".format(mem_usage(ad_feature_df)))
print("Memory Optimised Ad Feature dataframe:{0}".format(mem_usage(optimized_g2)))

# user_feature_df memory usage optimization
user_profile_df.info(memory_usage='deep')
optimized_g3 = user_profile_df.copy()
g3_int = user_profile_df.select_dtypes(include=['int'])
converted_int = g3_int.apply(pd.to_numeric,downcast='unsigned')
optimized_g3[converted_int.columns] = converted_int
g3_float = user_profile_df.select_dtypes(include=['float'])
converted_float = g3_float.apply(pd.to_numeric,downcast='float')
optimized_g3[converted_float.columns] = converted_float
print("Original User Feature dataframe:{0}".format(mem_usage(user_profile_df)))
print("Memory Optimised User Feature dataframe:{0}".format(mem_usage(optimized_g3)))

# Join to create training dataset
df1 = raw_sample_df_new.merge(optimized_g3, on="userid")
final_df = df1.merge(optimized_g2, on="adgroup_id")
final_df = final_df.rename(columns = {'new_user_class_level ': 'new_user_class_level'})
final_df.head()
# save original df
final_df_copy = copy.copy(final_df)

# Free up memory
df1 = None
raw_sample_df = None
raw_sample_df_new = None
ad_feature_df = None
user_profile_df = None
optimized_g1 = None
optimized_g2 = None
optimized_g3 = None

# Data processing
final_df['hist_cate_id'] = final_df['cate_id']
final_df['hist_adgroup_id'] = final_df['adgroup_id']

sparse_features = ['userid', 'adgroup_id', 'final_gender_code', 'age_level',
                   'pvalue_level', 'new_user_class_level',
                   'shopping_level', 'occupation',
                   'pid', 'cate_id', 'customer', 'campaign_id', 'brand']
sparse_features

dense_features = ['price']
dense_features

sequence_features = [feat for feat in final_df.columns if feat not in
                     ['userid', 'time_stamp', 'adgroup_id', 'pid', 'nonclk', 'clk','cms_segid', 'cms_group_id',
                      'final_gender_code', 'age_level','pvalue_level', 'shopping_level', 'occupation',
                      'new_user_class_level','cate_id', 'campaign_id', 'customer', 'brand', 'price']]
sequence_features


behavior_feature_list = ['cate_id', 'adgroup_id']
behavior_feature_list

final_df[sparse_features] = final_df[sparse_features].fillna(-1)
final_df[sequence_features] = final_df[sequence_features].fillna(0)
final_df[dense_features] = final_df[dense_features].fillna(0)
final_df['new_user_class_level'] = final_df['new_user_class_level'].astype('uint32')
final_df['pvalue_level'] = final_df['pvalue_level'].astype('uint32')
final_df['brand'] = final_df['brand'].astype('uint32')

for feat in sparse_features:
    lbe = LabelEncoder()
    final_df[feat] = lbe.fit_transform(final_df[feat])


for feat in sequence_features:
    lbe = LabelEncoder()
    final_df[feat] = lbe.fit_transform(final_df[feat])


mms = MinMaxScaler(feature_range=(0, 1))
final_df[dense_features] = mms.fit_transform(final_df[dense_features])

train, test = train_test_split(final_df, test_size=0.2, random_state = 42)
target = ['clk']

# Training model to calculate pClick_private
def convert(name, df):
    if name[:5] == 'hist_':
        return np.array([[item] for item in df[name].to_numpy()])
    else:
        return df[name].to_numpy()

sparse_features = ['userid', 'adgroup_id', 'final_gender_code', 'cate_id']
dense_features = ['price']
sequence_features = ['hist_cate_id', 'hist_adgroup_id']
behavior_feature_list = ['cate_id', 'adgroup_id']

feature_columns = \
    [SparseFeat(feat, vocabulary_size=final_df[feat].nunique(), embedding_dim=8) for feat in sparse_features] +\
    [DenseFeat(feat, 1, )for feat in dense_features]

feature_columns += \
    [VarLenSparseFeat(SparseFeat('hist_cate_id', vocabulary_size=final_df['hist_cate_id'].nunique(), embedding_dim=8,
                                 embedding_name='cate_id'), maxlen=1),
     VarLenSparseFeat(SparseFeat('hist_adgroup_id', vocabulary_size=final_df['hist_adgroup_id'].nunique(), embedding_dim=8,
                                 embedding_name='adgroup_id'), maxlen=1)]

linear_feature_columns = feature_columns
dnn_feature_columns = feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )

train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}
all_model_input = {name:final_df[name] for name in feature_names}

model_private = DIN(linear_feature_columns, behavior_feature_list, dnn_use_bn=False, dnn_hidden_units=(200, 80),
                    dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",
                    att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0,
                    seed=1024, task='binary')

# model_private = DIN(linear_feature_columns, behavior_feature_list)
model_private.compile("adam", "binary_crossentropy",metrics=['binary_crossentropy', 'AUC'], )
history = model_private.fit(train_model_input, train[target].values,
                            batch_size=5024, epochs=5, verbose=1, validation_split=0.25, )

pred_ans_private = model_private.predict(all_model_input, batch_size=256)

# Training model to calculate pClick_Public
# Remove private user features
sparse_features = ['adgroup_id', 'pid', 'cate_id', 'customer', 'campaign_id', 'brand']

fixlen_feature_columns = \
    [SparseFeat(feat, vocabulary_size=final_df[feat].nunique(), embedding_dim=8) for feat in sparse_features] +\
    [DenseFeat(feat, 1, )for feat in dense_features]+\
    [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=final_df[feat].nunique(), embedding_dim=8), maxlen=1) for feat in sequence_features]

linear_feature_columns = fixlen_feature_columns
dnn_feature_columns = fixlen_feature_columns
feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )

train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}
all_model_input = {name:final_df[name] for name in feature_names}

model_public = FNN(linear_feature_columns, dnn_feature_columns)
model_public.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy', 'AUC'], )
history = model_public.fit(train_model_input, train[target].values,
                           batch_size=5024, epochs=5, verbose=1, validation_split=0.25, )

pred_ans_public = model_public.predict(all_model_input, batch_size=256)

# Create pClick_Private and pClick_Public column
final_df = copy.copy(final_df_copy)
final_df['pclick_private'] = pred_ans_private
final_df['pclick_public'] = pred_ans_public

(
        final_df[['userid', 'time_stamp', 'adgroup_id', 'price', 'clk', 'pclick_private', 'pclick_public']]
        .to_csv(f'{args.data_path}/all_data_with_pclick_prediction.csv', index=False)
)

