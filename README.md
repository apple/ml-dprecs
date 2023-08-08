# Differentially-private, precisely-measureable, personalized recommendations (DPRECS)

This software project accompanies the research paper, [Randomized algorithms for precise measurement of differentially-private, personalized recommendations](https://arxiv.org/abs/2308.03735).

In this work, we propose an algorithm for personalized recommendations that respects the key requirements of user privacy and precise measurement. 
We consider personalized advertising as an example application, and conduct experiments on real digital ads auctions data to quantify how the 
proposed privacy-preserving algorithm affects key metrics related to user experience, advertiser value, and platform revenue compared to the 
extremes of both (private) non-personalized and non-private, personalized implementations. 

The code here should facilitate reproduction of the results on the external dataset (display ads from Alibaba).
The notebooks TaobaoAnalysis.ipynb can be used to re-run the analysis and plot the figures shown in the paper.


## Getting Started 
### Environment
This code is designed to work with Python 3.8.
Conda is recommended.
Use pip to install dependencies:
```
pip install ."[dev]"
```

To verify environment setup, run the unit tests:
```
python -m unittest
```
### Configs
Change the TABLE_NAME and RESULTS_TABLE_NAME configs in dprecs/taobao/configs_X.py to be writable table names 
(TABLE_NAME will be used to write the "raw" data; RESULTS_TABLE_NAME will store the results of the auctions). 

### Producing pClick_Private and pClick_Public
In order to obtain pClick models for our experiment, we refer to following paper to generate estimated Click-through rate for a certain ads:
```
@inproceedings{zhou2018deep,
  title={Deep interest network for click-through rate prediction},
  author={Zhou, Guorui and Zhu, Xiaoqiang and Song, Chenru and Fan, Ying and Zhu, Han and Ma, Xiao and Yan, Yanghui and Jin, Junqi and Li, Han and Gai, Kun},
  booktitle={Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery \& data mining},
  pages={1059--1068},
  year={2018}
}
```

We start from separate tables (raw_samples, ad_feature, user_profile, etc.) in Alibaba Dataset and join them altogether on unique key "adgroup_id" to collect all statistics for a certain ads. We followed the same definition in DeepCTR to categorize features into different types:
```
@misc{shen2017deepctr,
  author = {Weichen Shen},
  title = {DeepCTR: Easy-to-use,Modular and Extendible package of deep-learning based CTR models},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/shenweichen/deepctr}},
}
```
where we have sparse_features to containing categorical features from the feature set (e.g. shopping category), dense_features containing float features from the feature set (e.g. item price), sequence_features containing historical features from the feature set (e.g. user behaviors in a certain time window). The prediction target is whether user would click on this ads.

Our model uses Deep Interest Network described in the paper with following paramters:
model = DIN(linear_feature_columns, behavior_feature_list, dnn_use_bn=True,dnn_hidden_units=(200, 80), dnn_activation='relu', att_hidden_size=(80, 40), att_activation="dice",att_weight_normalization=False, l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, seed=1024,task='binary')

Training will optimize binary_crossentropy with Adam optimizer.

Specifically, to generate pClick_Public, we intend to remove all features collected from user_profile table. To keep models with same input structure, we manually masked out all features including age, gender, occupation, shopping level, etc. and train using the same model structure.

To do this, download the taobao data from the source (https://tianchi.aliyun.com/dataset/56) to a local DATA_DIR, then run:
```
python dprecs/taobao/get_deepctr_predictions.py --data_path DATA_DIR
```
This should yield a new file, `all_data_with_pclick_prediction.csv` with the relevant data for the rest of the analysis.

### Reproducing results on Taobao dataset

Open `ml-dprecs/notebooks/TaobaoAnalysis.ipynb`. Specify the location of the new data file with pClick estimates (e.g. `all_data_with_pclick_prediction.csv`)
and re-run all cells. 

## Reference

[1] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li and Kun Gai. Deep interest network for click-through rate prediction. Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery \& data mining, 2018.

## Citation

If you find this code useful in your research, please cite:
```
@inproceedings{dprecs,
  title={Randomized algorithms for precise measurement of differentially-private, personalized recommendations},
  author={Laro, Allegra and Chen, Yanqing and He, Hao and Aghazadeh, Babak },
  booktitle={TBD},
  year={2023}
}
```
