import pandas as pd
import numpy as np
from tweepy import API
from tweepy.models import User
from data_models.twitter_models import StatusDoc
from connectors.twitter_api_connector import TwitterConnector
from analysis.feature_extraction import FeatureExtraction
from typing import Optional
from helper.settings import (logger, popularity_metric, boolean_cols,
                             drop_num_cols, id_col, scaler_filename,
                             document_col, cat_cols, link_information,
                             default_value, join_attr)


class UserAccountAnalysis:
    def __init__(self, consumer_key: str, consumer_secret: str,
                 access_token: str, access_token_secret: str):
        self.twitter_api_connector: TwitterConnector = TwitterConnector(
            consumer_key=consumer_key,
            consumer_secret=consumer_secret,
            access_token=access_token,
            access_token_secret=access_token_secret)
        self.api: Optional[API] = None

    def set_up_twitter_api(self):
        if self.twitter_api_connector.api is None:
            self.twitter_api_connector.set_up_twitter_api_connection()
        self.api: API = self.twitter_api_connector.api

    def get_users_data_by_screen_name(self, screen_names: list) -> list:
        users_data: list = []
        try:
            # Connect API
            self.set_up_twitter_api()

            # Retrieve data
            users_data: list = self.twitter_api_connector.get_user_profiles_by_screen_names(
                api=self.api, screen_names=screen_names)
        except Exception as e:
            logger.error(e)
        return users_data

    def get_users_data_by_id(self, user_ids: list) -> list:
        users_data: list = []
        try:
            # Connect API
            self.set_up_twitter_api()

            # Retrieve data
            users_data: list = self.twitter_api_connector.get_user_profiles_by_ids(
                api=self.api, user_ids=user_ids)
        except Exception as e:
            logger.error(e)
        return users_data

    def generate_twitter_account_database_by_user_ids(self, user_ids: list, extended_info: bool = False) -> iter:
        user_data: iter = ([])
        try:
            # Connect API
            self.set_up_twitter_api()

            # Retrieve data
            user_data: iter = iter(self.twitter_api_connector.generate_twitter_account_database(
                api=self.api, user_ids=user_ids, extended_info=extended_info))

        except Exception as e:
            logger.error(e)
        return user_data

    def get_last_k_tweets_from_user_account(self, user_id: int, k: int = 10):
        last_k_tweets: iter = iter([])
        try:
            # Connect API
            self.set_up_twitter_api()

            # Retrieve data
            last_k_status: iter = self.twitter_api_connector.get_last_k_tweets_from_user_account(
                api=self.api, user_id=user_id, k=k)
            last_k_tweets: iter = iter([StatusDoc(status=status) for status in last_k_status])

        except Exception as e:
            logger.error(e)
        return last_k_tweets

    def extract_features_from_user_account(self, user_data: User):
        user_features: dict = {}
        try:
            # Retrieve user data
            user_features: dict = self.twitter_api_connector.retrieve_user_data(
                user_data)
        except Exception as e:
            logger.error(e)
        return user_features

    @staticmethod
    def preprocess_data(user_features: dict, popularity_metric: str, boolean_cols: list,
                        drop_num_cols: list, scaler_filename: str, cat_cols: list,
                        default_value: str, link_information: list, document_col: str,
                        join_attr: str = ". ", index_val: int = 0):

        data_transformed: pd.DataFrame = pd.DataFrame([])
        try:
            data_transformed: pd.DataFrame = FeatureExtraction.preprocess_prediction_data(
                data_dct=user_features, popularity_metric=popularity_metric,
                boolean_cols=boolean_cols,
                drop_num_cols=drop_num_cols,
                scaler_filename=scaler_filename,
                cat_cols=cat_cols,
                default_value=default_value,
                link_information=link_information,
                document_col=document_col,
                join_attr=join_attr,
                index_val=index_val)
        except Exception as e:
            logger.error(e)
        return data_transformed

    @staticmethod
    def generate_embedding(data: pd.DataFrame, document_col: str, id_col: str, embeddings: list,
                           doc2vec: str = "transformer_roberta"):
        doc_embedding_np: np.ndarray = np.array([])
        try:
            all_num_cols: list = FeatureExtraction.get_numerical_columns(data=data)
            all_num_cols.remove(id_col)

            document: str = data[document_col]

            # 3.1 Generate doc embedding
            doc_emb: np.ndarray = FeatureExtraction.generate_doc_embedding(document=document,
                                                                           embeddings=embeddings,
                                                                           doc2vec=doc2vec)
            x_doc_emb: list = list(doc_emb.tolist())
            x_num: list = [data[j] for j in all_num_cols]

            if len(x_doc_emb) > 0:
                # 3.3 Concatenate doc embedding + numerical cols
                doc_embedding_np: np.ndarray = np.array([x_num + x_doc_emb]).reshape((1, -1))
                print(doc_embedding_np.shape)
        except Exception as e:
            logger.error(e)
        return doc_embedding_np

    def generate_user_feature_vector(self, user: User) -> np.ndarray:
        doc_emb: np.ndarray = np.array([])
        try:
            index_val: int = 0

            # 1. Extract features
            user_features: dict = self.extract_features_from_user_account(
                user_data=user)

            # 2. Preprocess features
            data_transformed: pd.DataFrame = self.preprocess_data(
                user_features=user_features,
                popularity_metric=popularity_metric,
                boolean_cols=boolean_cols,
                drop_num_cols=drop_num_cols,
                scaler_filename=scaler_filename,
                cat_cols=cat_cols,
                default_value=default_value,
                link_information=link_information,
                document_col=document_col,
                join_attr=join_attr,
                index_val=index_val)

            # 3. Generate input embedding
            doc_emb: np.ndarray = self.get_embedding_from_dataframe(
                data_transformed=data_transformed, index_val=index_val)

        except Exception as e:
            logger.error(e)
        return doc_emb

    @staticmethod
    def get_embedding_from_dataframe(data_transformed: pd.DataFrame, index_val: int = 0):
        doc_emb: np.ndarray = np.array([])
        try:
            # Generate input embedding
            all_num_cols: list = FeatureExtraction.get_numerical_columns(
                data=data_transformed)
            all_num_cols.remove(id_col)

            doc_emb: np.ndarray = FeatureExtraction.get_doc2vec_embedding(
                data_input=data_transformed,
                index_val=index_val,
                document_col=document_col,
                all_num_cols=all_num_cols)
        except Exception as e:
            logger.error(e)
        return doc_emb