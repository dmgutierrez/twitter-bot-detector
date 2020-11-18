# User2Vec

A Transformed-based approach for encoding Twitter user Accounts. The model can be used to generate 256-dimensional embeddings from an input user account. This encoding can be employed in any Information Retrieval (IR) application.

```python
import numpy as np
from api.user_account_api import UserAccountAPI

# You need to add your Twitter API Credentials
CONSUMER_KEY = "xxx"
CONSUMER_SECRET = "xxx"
ACCESS_TOKEN = "xxx"
ACCESS_TOKEN_SECRET = "xx"

# 1. Set up the API object
api: UserAccountAPI = UserAccountAPI(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET)

# 2. Get Feature Input vectors using a Transformer model

# You retrieve the user account and generate the input vector 
# either by screen name or by the user id 

screen_name: str = "@poem_exe"
user_input_feature_vec1: np.ndarray = api.get_input_feature_vector_by_screen_name(
    screen_name=screen_name)

user_id: str = "2579497028"
user_input_feature_vec2: np.ndarray = api.get_input_feature_vector_by_id(
    user_id=user_id)

# ---------------------------------------------------
# 3. Predict credibility

credibility1: float = api.get_user_account_credibility(
    input_user_embedding=user_input_feature_vec1)
print(credibility1)

credibility2: float = api.get_user_account_credibility(
    input_user_embedding=user_input_feature_vec2)
print(credibility2)

# ---------------------------------------------------
# 4. Get embedding
user_embedding1: np.ndarray = api.get_user_embedding(
    input_user_embedding=user_input_feature_vec1)
print(user_embedding1.shape)

user_embedding2: np.ndarray = api.get_user_embedding(
    input_user_embedding=user_input_feature_vec2)
print(user_embedding2.shape)
```

