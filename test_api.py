import numpy as np
from api.user_account_api import UserAccountAPI

CONSUMER_KEY = "WcIJMyBF3spFEtRGHwqDlAKDT"
CONSUMER_SECRET = "IfNlsgNrW5o6rfhayOURF1BcUDBcsrfgqss9g5inqeFytDbkgE"
ACCESS_TOKEN = "973174862159253505-mAYpqjzegRXFNhdEPXOkDLsHWXePp7q"
ACCESS_TOKEN_SECRET = "0mmfQcNDJBIFhMM9bexZSO1eIKhFdaP8JX9cMnBG81gJE"

api: UserAccountAPI = UserAccountAPI(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET)

# Get Feature vectors

screen_name: str = "@cnn"
user_input_feature_vec1: np.ndarray = api.get_input_feature_vector_by_screen_name(
    screen_name=screen_name)

user_id: str = "2579497028"
user_input_feature_vec2: np.ndarray = api.get_input_feature_vector_by_id(
    user_id=user_id)


# 1. Predict credibility

credibility1: float = api.get_user_account_credibility(
    input_user_embedding=user_input_feature_vec1)
print(credibility1)

credibility2: float = api.get_user_account_credibility(
    input_user_embedding=user_input_feature_vec2)
print(credibility2)

# 2. Get embedding
user_embedding1: np.ndarray = api.get_user_embedding(
    input_user_embedding=user_input_feature_vec1)
print(user_embedding1.shape)

user_embedding2: np.ndarray = api.get_user_embedding(
    input_user_embedding=user_input_feature_vec2)
print(user_embedding2.shape)

