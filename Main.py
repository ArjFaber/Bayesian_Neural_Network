import numpy as np

from Data import create_data_set
from Random_Search import random_search_BNN

weather_ds = create_data_set()
X = np.asarray(weather_ds[['weather_drizzle','weather_fog', 'weather_rain', 'weather_snow', 'weather_sun','precipitation', 'wind', 'temp_min']])
y = np.asarray(weather_ds['temp_max'])
X = X.astype('float32')
y = y.astype('float32')

#Number of nodes in the hidden layer:
hl_nodes = [int(x) for x in np.linspace(10,50, num = 10)]
#Number of learning cycles:
epochs = [int(x) for x in np.linspace(50,1000, num = 10)]
#learning_rate
learning_rate = [float(x) for x in np.linspace(0.01,0.0001, num = 3)] 
random_grid = {'hl_nodes': hl_nodes,
               'epochs' : epochs,
               'learning_rate': learning_rate
               }
print(random_grid)

random_search_BNN(hl_nodes,epochs,learning_rate, X, y)