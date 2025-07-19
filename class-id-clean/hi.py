import pickle

with open('./cache/session_1_raw/000015.pickle', 'rb') as f:
    data = pickle.load(f)

second_element = data[1]


print((second_element[0].keys()))