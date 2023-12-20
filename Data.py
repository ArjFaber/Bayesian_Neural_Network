import pandas as pd

def create_data_set():
    weather_ds = pd.read_csv("seattle-weather.csv")
    weather_ds.describe()
    weather_ds = weather_ds
    weather_ds_copy = weather_ds
    data_dummies = weather_ds_copy.drop(columns=['date', 'precipitation', 'temp_max', 'temp_min', 'wind' ])
    data_dummies = pd.get_dummies(data=data_dummies).astype(int)
    weather_ds_copy = weather_ds_copy[['precipitation', 'wind', 'temp_min', 'temp_max']]
    weather_ds = pd.concat([data_dummies,weather_ds_copy], axis = 1)
    weather_ds.head()
    return weather_ds
