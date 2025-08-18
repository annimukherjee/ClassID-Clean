import pickle
import pandas as pd

with open("./output/1_tracking_results.pkl", "rb") as f:
    data = pickle.load(f)

# Show all rows
pd.set_option('display.max_rows', None)

print(data)