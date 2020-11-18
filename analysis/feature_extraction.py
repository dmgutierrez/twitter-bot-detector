import pandas as pd
import numpy as np
import joblib, os
import flair
from flair.data import Sentence
from torch import Tensor
from helper.utils import prepare_directory
from helper.settings import logger
from sklearn.preprocessing import StandardScaler
from typing import Optional
from flair.embeddings import (FlairEmbeddings, DocumentPoolEmbeddings,
                              ELMoEmbeddings, BertEmbeddings,
                              TransformerDocumentEmbeddings,
                              DocumentRNNEmbeddings)


class FeatureExtraction(object):

    @staticmethod
    def generate_doc_embedding(document: str, embeddings: list, doc2vec="transformer_roberta"):
        doc_embedding: np.ndarray = np.array([])
        try:
            logger.info("Generating embedding for document .... ")
            # 1. Initialise Document Embedding

            # a) Pooling
            if doc2vec == "pool":
                document_embeddings: DocumentPoolEmbeddings = DocumentPoolEmbeddings(
                    embeddings=embeddings)
            elif doc2vec == "rnn":
                document_embeddings: DocumentRNNEmbeddings = DocumentRNNEmbeddings(
                    embeddings=embeddings,
                    hidden_size=256,
                    rnn_type='LSTM')

            # b) Transformer
            elif doc2vec == "transformer_bert":
                document_embeddings: TransformerDocumentEmbeddings = TransformerDocumentEmbeddings (
                    'bert-base-multilingual-cased')
            else:
                document_embeddings: TransformerDocumentEmbeddings = TransformerDocumentEmbeddings(
                    'roberta-base')

            # 2. Create an example sentence
            sentence: Sentence = Sentence(document)

            # 3. Embed the sentence with our document embedding
            document_embeddings.embed(sentence)

            # 4. Save embedding into CPU
            if "cuda" in str(flair.device).lower():
                doc_emb_cpu: Tensor = sentence.embedding.cpu()
                # 5. Convert to numpy array
                doc_embedding: np.ndarray = doc_emb_cpu.detach().numpy()
            else:
                doc_embedding: np.ndarray = sentence.get_embedding().detach().numpy()
        except Exception as e:
            logger.error(e)
        return doc_embedding

    @staticmethod
    def set_up_flair_cpu_device():
        flair.device = "cpu"
        logger.info(f"Flair device: {flair.device}")

    @staticmethod
    def generate_document_from_features(text_features: list, link_information: list,
                                        join_attr: str = ". "):
        document: str = ""
        try:
            document_lst: list = [link + " : " + text for text, link in zip(text_features,
                                                                            link_information)]
            document: str = join_attr.join(document_lst)
        except Exception as e:
            logger.error(e)
        return document

    @staticmethod
    def extract_document_from_categorical_series(row: pd.Series, link_information: list,
                                                 join_attr: str = ". "):
        document: str = ""
        try:
            text_features: list = [row[col] for col in list(row.index)]
            document: str = FeatureExtraction.generate_document_from_features(
                text_features=text_features,
                link_information=link_information,
                join_attr=join_attr)
        except Exception as e:
            print(e)
        return document

    @staticmethod
    def convert_bool_to_int(data: pd.DataFrame, boolean_cols: list):
        try:
            for col in boolean_cols:
                data[col] = data[col].astype(int)
        except Exception as e:
            logger.error(e)
        return data

    @staticmethod
    def popularity_metric(friends_count: int, followers_count: int):
        return np.round(np.log(1 + friends_count) * np.log(1 + followers_count), 3)

    @staticmethod
    def compute_popularity_metric(row, col_friends_count: str = "friends_count",
                                  col_followers_count: str = "followers_count"):
        return FeatureExtraction.popularity_metric(
            friends_count=row[col_friends_count],
            followers_count=row[col_followers_count])

    @staticmethod
    def preprocess_dataframe_train(data: pd.DataFrame, popularity_metric: str, boolean_cols: list,
                                   drop_num_cols: list, scaler_filename: str, cat_cols: list,
                                   default_value: str, link_information: list, document_col: str,
                                   target_col: str, join_attr: str = ". "):

        data_transformed: pd.DataFrame = pd.DataFrame([])
        try:
            # Save target variable
            data_target: pd.DataFrame = data[[target_col]]

            # Convert to int boolean
            data: pd.DataFrame = FeatureExtraction.convert_bool_to_int(
                data=data, boolean_cols=boolean_cols)

            # Extract popularity metric
            data[popularity_metric] = data.apply(FeatureExtraction.compute_popularity_metric,
                                                 axis=1)

            df_features: pd.DataFrame = data.copy()

            logger.info("Preprocessing Numerical Columns")
            # Preprocess Numerical Features
            data_transformed_num: pd.DataFrame = FeatureExtraction.preprocess_numerical_data(
                data=df_features,
                drop_cols=drop_num_cols,
                scaler_filename=scaler_filename)

            logger.info("Preprocessing Categorical Columns")
            # Preprocess Categorical Features
            data_transformed_cat: pd.DataFrame = FeatureExtraction.preprocess_categorical_data(
                data=df_features,
                cat_cols=cat_cols,
                default_value=default_value,
                link_information=link_information,
                document_col=document_col,
                join_attr=join_attr)

            logger.info("Concatenating Transformed Columns")
            # Concatenate DataFrames
            data_transformed: pd.DataFrame = pd.concat([data_transformed_num,
                                                        data_transformed_cat,
                                                        data_target],
                                                       axis=1, sort=False)

        except Exception as e:
            logger.error(e)
        return data_transformed

    @staticmethod
    def get_numerical_columns(data: pd.DataFrame):
        return list(data._get_numeric_data().columns)

    @staticmethod
    def preprocess_numerical_data(data: pd.DataFrame, drop_cols: list, scaler_filename: str,
                                  fit=True):
        data_transformed: pd.DataFrame = pd.DataFrame([])
        try:
            # Extract Numerical Features
            df_num: pd.DataFrame = data._get_numeric_data()
            df_num.drop(drop_cols, axis=1, inplace=True)

            # Scale data
            data_transformed: pd.DataFrame = FeatureExtraction.scale_data(
                data=df_num, fit=fit, filename=scaler_filename)

            # Add drop cols
            data_transformed: pd.DataFrame = pd.concat([data[drop_cols],
                                                        data_transformed], axis=1, sort=False)

        except Exception as e:
            logger.error(e)
        return data_transformed

    @staticmethod
    def preprocess_categorical_data(data: pd.DataFrame, cat_cols: list, default_value: str,
                                    link_information: list, document_col: str,
                                    join_attr: str = ". "):
        data_transformed: pd.DataFrame = pd.DataFrame([])
        try:
            df_cat: pd.DataFrame = data.select_dtypes(["object"])
            df_cat: pd.DataFrame = df_cat[cat_cols]
            df_cat.fillna(value=default_value, inplace=True)

            # Add document
            df_cat[document_col] = df_cat.apply(FeatureExtraction.extract_document_from_categorical_series,
                                                axis=1,
                                                args=(link_information, join_attr,))

            data_transformed: pd.DataFrame = df_cat.copy()

        except Exception as e:
            logger.error(e)
        return data_transformed

    @staticmethod
    def scale_data(data: pd.DataFrame, fit: bool = True, filename: str = ""):
        data_transformed: pd.DataFrame = pd.DataFrame([])
        try:
            if fit:
                scaler: StandardScaler = StandardScaler()
                scaler.fit(data)

                prepare_directory(os.sep.join(filename.split(os.sep)[0:-1]))
                # Save scaler
                logger.info("Saving scaler object at %s", filename)

                FeatureExtraction.save_scaler_object(scaler_obj=scaler, filename=filename)
            else:

                scaler: StandardScaler = FeatureExtraction.load_scaler_object(
                    filename=filename)

            # Transform data
            res_transformed: np.ndarray = scaler.transform(data)
            data_transformed: pd.DataFrame = pd.DataFrame(res_transformed,
                                                          columns=data.columns)
        except Exception as e:
            logger.error(e)
        return data_transformed

    @staticmethod
    def get_flair_embeddings():
        jw_forward: FlairEmbeddings = FlairEmbeddings("multi-forward", chars_per_chunk=128)
        jw_backward: FlairEmbeddings = FlairEmbeddings("multi-backward", chars_per_chunk=128)
        embeddings: list = [jw_forward, jw_backward]
        return embeddings

    @staticmethod
    def get_bert_embeddings():
        bert_embedding: BertEmbeddings = BertEmbeddings('bert-base-multilingual-cased')
        embeddings: list = [bert_embedding]
        return embeddings

    @staticmethod
    def get_elmo_embeddings():
        bert_embedding: ELMoEmbeddings = ELMoEmbeddings('small')
        embeddings: list = [bert_embedding]
        return embeddings

    @staticmethod
    def save_scaler_object(scaler_obj: StandardScaler,
                           filename: str):
        try:
            joblib.dump(scaler_obj, filename)
        except Exception as e:
            logger.error(e)

    @staticmethod
    def load_scaler_object(filename: str):
        scaler_obj: Optional[StandardScaler] = None
        try:
            scaler_obj: StandardScaler = joblib.load(filename)
        except Exception as e:
            logger.error(e)
        return scaler_obj

    @staticmethod
    def preprocess_prediction_data(data_dct: dict, popularity_metric: str, boolean_cols: list,
                                   drop_num_cols: list, scaler_filename: str, cat_cols: list,
                                   default_value: str, link_information: list, document_col: str,
                                   join_attr: str = ". ", index_val: int = 0):

        data_transformed: pd.DataFrame = pd.DataFrame([])
        try:
            data: pd.DataFrame = pd.DataFrame(data_dct, index=[index_val])

            # Convert to int boolean
            data: pd.DataFrame = FeatureExtraction.convert_bool_to_int(
                data=data, boolean_cols=boolean_cols)

            # Extract popularity metric
            data[popularity_metric] = data.apply(FeatureExtraction.compute_popularity_metric,
                                                 axis=1)

            df_features: pd.DataFrame = data.copy()

            logger.info("Preprocessing Numerical Columns")
            # Preprocess Numerical Features
            data_transformed_num: pd.DataFrame = FeatureExtraction.preprocess_numerical_data(
                data=df_features,
                drop_cols=drop_num_cols,
                scaler_filename=scaler_filename, fit=False)

            logger.info("Preprocessing Categorical Columns")
            # Preprocess Categorical Features
            data_transformed_cat: pd.DataFrame = FeatureExtraction.preprocess_categorical_data(
                data=df_features,
                cat_cols=cat_cols,
                default_value=default_value,
                link_information=link_information,
                document_col=document_col,
                join_attr=join_attr)

            logger.info("Concatenating Transformed Columns")
            # Concatenate DataFrames
            data_transformed: pd.DataFrame = pd.concat([data_transformed_num,
                                                        data_transformed_cat],
                                                       axis=1, sort=False)
        except Exception as e:
            logger.error(e)
        return data_transformed

    @staticmethod
    def get_doc2vec_embedding(data_input: Optional[pd.DataFrame],
                              index_val: int, document_col: str,
                              all_num_cols: list):
        doc_embedding_np: np.ndarray = np.array([])
        try:
            if isinstance(data_input, pd.DataFrame):
                document: str = data_input.loc[index_val, document_col]
                x_num: list = [data_input.loc[index_val, j] for j in all_num_cols]
            elif isinstance(data_input, pd.Series):
                document: str = data_input[document_col]
                x_num: list = [data_input[j] for j in all_num_cols]
            else:
                document: str = data_input.get(document_col, "")
                x_num: list = [data_input.get(document_col, -1) for j in all_num_cols]

            # Document embedding from a string
            doc_emb: np.ndarray = FeatureExtraction.generate_doc_embedding(
                document=document,
                embeddings=FeatureExtraction.get_flair_embeddings())

            x_doc_emb: list = list(doc_emb.tolist())

            if len(x_doc_emb) > 0:
                # 3.3 Concatenate doc embedding + numerical cols
                doc_embedding_np: np.ndarray = np.array([x_num + x_doc_emb]).reshape((1, -1))
        except Exception as e:
            logger.error(e)
        return doc_embedding_np