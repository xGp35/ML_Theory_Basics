# Load the data
import dowhy
from dowhy import CausalModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt

data = dowhy.datasets.lalande_dataset()
print(data.head())

features = ['age', 'educ', 'black', 'hisp', 'married', 'nodegr', 're74', 're75']

X =  data[features]
y = data['re78']
T = data['treat']

#Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Define the causal model using dowhy
model = CausalModel(
    data = data,
    treatment = 'treat',
    outcome = 're78',
    common_causes = features
)

# Identify the causal effect
estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate the causal effect using a regression model
estimate = model.estimate_effect(
    identified_estimand=estimand,
    method_name="backdoor.ecoml.dml.LinearDML",
    target_units='ate',
    method_params={
        'init_params': {
            'model_y': LGBMRegressor(n_estimators = 100, max_depth=3, verbose=1),
            'model_t': LogisticRegression(max_iter=1000),
            'discrete_treatment' : True,
        },
        'fit_params' : {}
    }   
)

#Display the causal effect
print(f"Estimated Average Treatment Effect (ATE): {estimate.value}")

# 4. Refutation

# To verify the robustness of our causal model, we will use refutation methods. 
# In causal inference, it is crucial to check whether the assumed causal graph is robust. 
# We will use two methods:

# Random Common Cause: 
# This method adds a random variable to the dataset as a common cause.
# If the original estimate is robust, the causal estimate should not change significantly.

#Placebo Treatment: This method replaces the treatment variable with a random variable. 
# If the original estimate is valid, the refuted result should be close to zero.




refutation_methods = [
    "random_common_cause",
    "placebo_treatment_refuter"
]

for method in refutation_methods:
    result = model.refute_estimate(estimand, estimate, method_name=method)
    print(result)



