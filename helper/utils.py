import os, json
import numpy as np
import more_itertools as mit
from uuid import uuid4
from datetime import datetime
from collections import Counter
from helper.settings import logger
import pandas as pd
import hashlib


def preprocess_date(non_formatted_date: datetime,
                    new_format: str = '%Y-%m-%d %H:%M:%S') -> str:
    return non_formatted_date.strftime(new_format)


def get_average_tweets_per_day(statuses_count: int, created_at: datetime):
    average_tweets_per_day: float = 0.0
    try:
        account_age_days = get_account_age_in_days(created_at=created_at)
        average_tweets_per_day: float = float(np.round(statuses_count / account_age_days, 3))
    except Exception as e:
        logger.error(e)
    return average_tweets_per_day


def get_account_age_in_days(created_at: datetime):
    delta = datetime.utcnow() - created_at
    return delta.days


def int_to_str_list(data_ls: list):
    return [str(element) for element in data_ls]


def chunks_from_list(data_ls: list, n: int):
    """Yield successive n-sized chunks from lst."""
    split_lst: list = [list(el) for el in mit.divide(n, data_ls)]
    return split_lst


def prepare_directory(dir_path_to_check: str):
    res: bool = False
    try:
        if not os.path.exists(dir_path_to_check):
            os.makedirs(dir_path_to_check)
        res = True
    except Exception as e:
        logger.error(e)
    return res


def write_json_file(data: dict, filename: str):
    try:
        json_data = json.dumps(str(data))
        with open(filename, 'w') as file:
            json.dump(json_data, file)
    except Exception as e:
        logger.error(e)


def read_json_file(filename):
    data = None
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except Exception as e:
        logger.error(e)
    return data


def most_common(lst: list):
    data = Counter(lst)
    return max(lst, key=data.get)


def generate_uuid():
    event_id: str = str(uuid4())
    return event_id


def read_all_files_from_local_storage(local_storage_path: str, column_index: int, extension: str = ".csv") -> iter:
    data: iter = ([])
    try:
        # 1. Get all paths
        data_files: list = [f for f in os.listdir(local_storage_path) if f.endswith(extension)]

        data_list: list = []
        # 2. For loop over csv
        for data_file in data_files:
            data_file_path: str = os.path.join(local_storage_path, data_file)
            # 1. Load DataFrame
            df: pd.DataFrame = pd.read_csv(data_file_path, header=None)

            # 2. Retrieve columns
            data_list += df.iloc[:, column_index].values.flatten().tolist()
        data: iter = iter(data_list)
    except Exception as e:
        logger.error(e)
    return data


def generate_128_uuid_from_string(data_uuid: str) -> str:
    identifier: str = ""
    try:
        identifier: str = hashlib.sha512(data_uuid.lower().encode('utf-8')).hexdigest()
    except Exception as e:
        print(e)
    return identifier