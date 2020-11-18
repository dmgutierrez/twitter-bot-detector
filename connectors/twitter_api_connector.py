import time
import numpy as np
from tweepy import API, OAuthHandler, Cursor
from tweepy.models import User, ResultSet, Status
from tweepy.error import RateLimitError
from typing import Optional
from data_models.twitter_models import UserProfile, UserAccountDoc, StatusDoc
from helper.utils import int_to_str_list, chunks_from_list
from helper.settings import (logger)
from GetOldTweets3.manager import TweetCriteria, TweetManager


class TwitterConnector:
    def __init__(self, consumer_key: str, consumer_secret: str,
                 access_token: str, access_token_secret: str):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.maximum_items: int = 500
        self.api: Optional[API] = None
        self.tweet_criteria: TweetCriteria = TweetCriteria()

    def set_up_twitter_api_connection(self) -> API:
        api: API = object.__new__(API)
        try:
            logger.info("Connecting to Twitter API ... ")
            auth: OAuthHandler = OAuthHandler(self.consumer_key,
                                              self.consumer_secret)
            auth.set_access_token(self.access_token, self.access_token_secret)
            api: API = API(
                auth,
                wait_on_rate_limit=True,
                wait_on_rate_limit_notify=True)
            self.api: API = api
        except Exception as e:
            logger.error(e)
        return api

    def get_historical_tweets(self, date_since: str, date_until: str, query: str,
                              max_tweets: int, lang: str):
        tweets: list = []
        try:
            self.tweet_criteria.setSince(date_since).setUntil(date_until
                                                              ).setQuerySearch(
                query).setMaxTweets(max_tweets).setLang(lang)

            # 1. Retrieve list of Tweet objects
            tweets: list = TweetManager.getTweets(self.tweet_criteria)
            print(tweets)
        except Exception as e:
            logger.error(e)
        return tweets

    @staticmethod
    def tweet_gathering(api: API, query: str, date_since: str,
                        lang: str = 'en'):
        try:
            logger.info("Retrieving Tweets ... ")
            # Collect tweets
            tweets = Cursor(
                api.search,
                lang=lang,
                q=query,
                include_entities=True,
                monitor_rate_limit=True,
                wait_on_rate_limit_notify=True,
                wait_on_rate_limit=True,
                result_type="recent",
                tweet_mode='extended').items()
            while True:
                try:
                    tweet: Status = tweets.next()
                    print(tweet)
                    yield tweet
                except RateLimitError:
                    time.sleep(60 * 15)
                    continue
                except StopIteration:
                    break
        except Exception as e:
            logger.error(e)

    @staticmethod
    def get_user_profiles_by_ids(api: API, user_ids: list) -> list:
        max_iter: int = 100
        users_data: list = []
        try:
            if len(user_ids) > max_iter:
                n: int = int(np.ceil(len(user_ids)/max_iter))
                data_chunks: list = chunks_from_list(data_ls=user_ids, n=n)
                logger.info("Retrieve user profiles from Twitter API ... ")
                for i, data in enumerate(data_chunks):
                    partial_user_ids: list = data
                    response: ResultSet = api.lookup_users(user_ids=partial_user_ids)
                    users_data.append(response)
            else:
                response: ResultSet = api.lookup_users(user_ids=user_ids)
                users_data.append(response)
        except Exception as e:
            logger.error(e)
        return users_data

    @staticmethod
    def get_user_profiles_by_screen_names(api: API, screen_names: list) -> list:
        max_iter: int = 100
        users_data: list = []
        try:
            logger.info("Retrieve user profiles from Twitter API ... ")
            if len(screen_names) > max_iter:
                n: int = int(np.ceil(len(screen_names)/max_iter))
                data_chunks: list = chunks_from_list(data_ls=screen_names, n=n)
                for i, data in enumerate(data_chunks):
                    response: ResultSet = api.lookup_users(screen_names=data)
                    users_data.append(response)
            else:
                response: ResultSet = api.lookup_users(screen_names=screen_names)
                users_data.append(response)
        except Exception as e:
            logger.error(e)
        return users_data

    @staticmethod
    def retrieve_user_data(user: User):
        return UserProfile(user).__dict__

    @staticmethod
    def retrieve_extended_user_data(user: User):
        return UserAccountDoc(user).__dict__

    @staticmethod
    def preprocess_user_profile_list(user_ids: list):
        return int_to_str_list(data_ls=user_ids)

    @staticmethod
    def generate_twitter_account_database(user_ids: list, api: API, extended_info: bool = False):
        user_info: list = []
        try:
            user_ids: list = __class__.preprocess_user_profile_list(user_ids)
            response: list = __class__.get_user_profiles_by_ids(api=api,
                                                                user_ids=user_ids)
            logger.info("Extract user profile information")
            if not extended_info:
                user_info: list = [__class__.retrieve_user_data(user) for
                                   result in response for user in result]
            else:
                user_info: list = [__class__.retrieve_extended_user_data(user) for
                                   result in response for user in result]
        except Exception as e:
            logger.error(e)
        return user_info

    @staticmethod
    def get_last_k_tweets_from_user_account(api: API, user_id: int, k: int = 20):
        latest_tweets: iter = iter([])
        try:
            latest_tweets: iter = api.user_timeline(id=user_id, count=k)
        except Exception as e:
            logger.error(e)
        return latest_tweets

    @staticmethod
    def get_tweet_data_from_tweet_id(api: API, tweet_id: int):
        tweet_data: Optional[Status] = None
        try:
            tweet_data: Status = api.get_status(id=tweet_id, tweet_mode='extended')
        except Exception as e:
            logger.error(e)
        return tweet_data

    def batch_tweet_process(self, date_since: str, date_until: str, search_term: list,
                            lang: str = "en", max_tweets: int = 500):
        try:
            # 2. Infinite process
            offset: int = 0
            query: str = " OR ".join(search_term)

            while offset < max_tweets:
                print("Offset ", str(offset))
                tweets: list = self.get_historical_tweets(
                    date_since=date_since,
                    date_until=date_until,
                    query=query,
                    max_tweets=max_tweets,
                    lang=lang)
                print(tweets)
                yield tweets
                offset += self.maximum_items
        except Exception as e:
            logger.error(e)

    def process_historical_tweets_status(self, tweets: list):
        tweets_docs: list = []
        users_docs: list = []
        try:
            # 1. Get Status from identifier
            tweets_status = [self.get_tweet_data_from_tweet_id(
                api=self.api, tweet_id=int(i.id)) for i in tweets]

            # 2. Process data
            for status in tweets_status:
                tweets_docs.append(StatusDoc(status=status))
                users_docs.append(UserAccountDoc(user=status.__getattribute__("user")))
        except Exception as e:
            logger.error(e)
        return zip(tweets_docs, users_docs)

    def process_batch_historical_tweets(self, date_since: str, date_until: str, search_term: list,
                                        lang: str = "en", max_tweets: int = 500):
        try:

            # 1. Call searching function
            for tweets in self.batch_tweet_process(
                    date_since=date_since,
                    date_until=date_until,
                    search_term=search_term,
                    max_tweets=max_tweets,
                    lang=lang):
                print(tweets)
                res: zip = self.process_historical_tweets_status(
                    tweets=tweets)
                yield res

        except Exception as e:
            logger.error(e)

    @staticmethod
    def get_available_languages(api: API) -> list:
        languages: list = []
        try:
            languages: list = api.supported_languages()
        except Exception as e:
            logger.error(e)
        return languages