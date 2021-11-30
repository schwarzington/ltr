from learning2rank.rank import ListNet
from pprint import pprint
import numpy as np
from sklearn.metrics import accuracy_score

import csv

from sklearn.model_selection import GroupShuffleSplit

import pandas as pd

from matplotlib import pyplot as plt

def race_to_df(data):
    df = pd.DataFrame(columns=["avg_finish_rank", "variance_rank", "flame_rate_rank", "win_rate_rank", "weighted_ability_rank", "distance_preference_rank", "mean_speed_rank"], data=data)
    return df

def predict(model, df):
    return model.predict(df.loc[:, ~df.columns.isin(['race_id', 'horse_id'])])

df = pd.read_csv("out_1000_ranked.csv", sep='\t', encoding='utf-8-sig')


gss = GroupShuffleSplit(test_size=.40, n_splits=1, random_state = 3).split(df, groups=df['race_id'])

X_train_inds, X_test_inds = next(gss)

train_data= df.iloc[X_train_inds]
X_train = train_data.loc[:, ~train_data.columns.isin(['race_id','place', 'horse_id'])]
y_train = train_data.loc[:, train_data.columns.isin(['place'])]

groups = train_data.groupby('race_id').size().to_frame('size')['size'].to_numpy()

test_data= df.iloc[X_test_inds]

#We need to keep the id for later predictions
X_test = test_data.loc[:, ~test_data.columns.isin(['place'])]
y_test = test_data.loc[:, test_data.columns.isin(['race_id', 'place'])]

import xgboost as xgb

model = xgb.XGBRanker(  
    tree_method='gpu_hist',
    booster='gbtree',
    objective='rank:pairwise',
    random_state=42, 
    learning_rate=0.001,
    colsample_bytree=0.9, 
    eta=0.02, 
    max_depth=3, 
    n_estimators=100, 
    subsample=0.80 
    )
setattr(model, 'verbosity', 2)

model.fit(X_train, y_train, group=groups, verbose=True)

array = []

with open("out_1000_ranked_test.csv") as file_name:
    file_read = csv.reader(file_name, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
    array = list(file_read)
    
pprint(array)

#Hardcoding the HorseId's so I can see what place they are in..
horses = ['8585', '16263', '11178', '18094', '8919', '15502', '19187','9302', '16718', '17649', '17063', '15292']

#More hardcoding to line the grouping up.
race_id = ['002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV', '002H5aRV']

data = race_to_df(array)
data['race_id'] = race_id

#We are only doing one prediction at the end because I haven't had time to injest all the data with the horse_ids
predictions = (data.groupby('race_id').apply(lambda x: predict(model, x)))

array_form = predictions.to_frame().to_numpy().tolist()

array_form_array = np.array(array_form)

results = []
count = 0
for result in array_form_array[0,0]:
    results.append((horses[count], result))
    count = count + 1
    
results.sort(key=lambda x:x[1], reverse=False)

'''horse_id	race_id	avg_finish_rank	variance_rank	flame_rate_rank	win_rate_rank	weighted_ability_rank	distance_preference_rank	mean_speed_rank	place
8585	002H5aRV	2	3	1	2	2	1	2	1
16263	002H5aRV	11	5	8	9	1	12	5	2
11178	002H5aRV	8	4	6	2	4	9	3	3
18094	002H5aRV	4	10	10	9	9	10	1	4
8919	002H5aRV	3	7	2	5	3	4	4	5
15502	002H5aRV	6	2	3	2	5	2	6	6
19187	002H5aRV	9	11	10	9	10	11	10	7
9302	002H5aRV	12	12	10	9	7	8	12	8
16718	002H5aRV	5	8	5	7	8	3	7	9
17649	002H5aRV	1	9	3	1	6	5	7	10
17063	002H5aRV	10	1	9	6	12	7	11	11
15292	002H5aRV	7	6	7	8	11	6	9	12'''
#this is what the predicted races actual results are..


pprint(results) 
