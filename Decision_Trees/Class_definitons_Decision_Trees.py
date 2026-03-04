import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Any


data = pd.read_csv('../data/iris.csv')
le = LabelEncoder()
data['type'] = le.fit_transform(data['species'])
data = data.drop(columns=['species'])

# col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
print(data.head())

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value

from pydantic import BaseModel
class Node(BaseModel):
    feature_index: Optional[int]
    threshold: Optional[float]
    left: Optional["Node"]
    right: Optional["Node"]
    info_gain: Optional[float]
    value: Optional[Any]

Node.model_rebuild()

# At the time Python reads the class definition, the class Node is not fully defined yet, so Pydantic cannot resolve "Node" immediately.
# So Pydantic needs a second pass to resolve the forward references.
# That’s what model_rebuild() does.

# Even cleaner (Python 3.10+)
# Since Optional[T]  ==  T | None

from pydantic import BaseModel
from typing import Any

class Node(BaseModel):
    feature_index: int | None = None
    threshold: float | None = None
    left: "Node" | None = None
    right: "Node" | None = None
    info_gain: float | None = None
    value: Any | None = None

Node.model_rebuild()


node = Node(
    feature_index=0,
    threshold=2.5,
    info_gain=0.41
)

# One Important ML Design Note

# Many implementations do NOT use Pydantic for tree nodes because:
# Trees can have thousands of nodes
# Pydantic adds validation overhead
# Simple classes are faster
# So typical ML code uses:
# class Node:
# instead of BaseModel.