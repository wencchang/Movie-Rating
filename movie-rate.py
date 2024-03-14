#%% 導入
import pandas as pd
import numpy as np  
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot
from keras.optimizers import Adam

#%% 讀取並打亂資料
data = pd.read_csv("C:/Users/User/Desktop/ITM/DataMining/HW5-movieRatingAssignment/movieRating.csv")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

#%% 重新編碼 UserID 和 MovieID 使其連續並從0開始
user_ids = data['UserID'].unique()
movie_ids = data['MovieID'].unique()
user_id_map = {id: i for i, id in enumerate(user_ids)}
movie_id_map = {id: i for i, id in enumerate(movie_ids)}
data['UserID'] = data['UserID'].map(user_id_map)
data['MovieID'] = data['MovieID'].map(movie_id_map) 

#%% 亂數拆成訓練集(80%)與訓練集(20%)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#%% 建立矩陣分解模型
num_users = len(user_ids)
num_movies = len(movie_ids)
user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')

# Embedding
user_embedding = Embedding(num_users, 8, name='user_embedding')(user_input)
movie_embedding = Embedding(num_movies, 8, name='movie_embedding')(movie_input)

# Flatten
user_vec = Flatten(name='flatten_users')(user_embedding)
movie_vec = Flatten(name='flatten_movies')(movie_embedding)

# DotProduct來預測評分
rating_prediction = Dot(name='rating_prediction', axes=1)([user_vec, movie_vec])

#%% 建立模型
model = Model(inputs=[user_input, movie_input], outputs=rating_prediction)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

#%% 訓練模型
history = model.fit([train_data['UserID'], train_data['MovieID']], train_data['Rating'], 
                    batch_size=64, epochs=20, verbose=1)

#%% 產出預測結果
predictions = model.predict([test_data['UserID'], test_data['MovieID']])
predictions = predictions.flatten()

#%% 計算 MAE，將 test_data['Rating'] 轉換為一維 Numpy 數組
true_ratings = test_data['Rating'].values
mae = np.mean(np.abs(predictions - true_ratings))
print(f"MAE: {mae}")
