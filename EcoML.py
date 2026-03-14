# Load the data
import dowhy
from sklearn.scaler import StandardScaler

data = dowhy.datasets.lalande_dataset()
print(data.head())

features = ['age', 'educ', 'black', 'hisp', 'married', 'nodegr', 're74', 're75']

X =  data[features]
y = data['re78']
T = data['treat']

#Scale the features
scaler = StandardScaler()
