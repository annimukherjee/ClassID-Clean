import pickle
import pandas as pd

with open("./output/4_features_extracted.pkl", "rb") as f:
    data = pickle.load(f)

# Show all rows
pd.set_option('display.max_rows', None)

print(data.shape)
print(data.columns)
print(data[['frame_id', 'track_id', 'face_embedding']])


print(data.iloc[0:4]['face_embedding']) 

print(len(data.iloc[0]['face_embedding']))