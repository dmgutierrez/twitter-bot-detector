from tweepy.models import User, Status
from twint.tweet import tweet
from twint.user import User as TwintUser
from datetime import datetime
from helper.utils import (preprocess_date, get_average_tweets_per_day,
                          get_account_age_in_days, generate_128_uuid_from_string)
from langdetect import detect


class TwitterDocument(object):
    @staticmethod
    def get_language(text: str):
        lang: str = "N/A"
        try:
            lang: str = detect(text=text)
        except Exception as e:
            pass
        return lang

    @staticmethod
    def get_128_uuid(data_str: str):
        return generate_128_uuid_from_string(data_uuid=data_str)


class UserProfile(TwitterDocument):
    def __init__(self, user: User):
        self.created_at: str = preprocess_date(user.__getattribute__(
            "created_at"))
        self.default_profile: bool = user.__getattribute__(
            "default_profile")
        self.default_profile_image: bool = user.__getattribute__(
            "default_profile_image")
        self.description: str = user.__getattribute__(
            "description")
        self.favourites_count: int = user.__getattribute__(
            "favourites_count")
        self.followers_count: int = user.__getattribute__(
            "followers_count")
        self.friends_count: int = user.__getattribute__(
            "friends_count")
        self.geo_enabled: bool = user.__getattribute__(
            "geo_enabled")
        self.id: int = user.__getattribute__("id")
        self.lang: str = self.get_language(text=user.__getattribute__("description"))
        self.location: str = "N/A" if user.__getattribute__(
            "location") is None or user.__getattribute__(
            "location") == "" else user.__getattribute__("location")
        self.profile_background_image_url: str = user.__getattribute__(
            "profile_background_image_url")
        self.profile_image_url: str = user.__getattribute__(
            "profile_image_url")
        self.screen_name: str = user.__getattribute__(
            "screen_name")
        self.statuses_count: int = user.__getattribute__(
            "statuses_count")
        self.verified: bool = user.__getattribute__(
            "verified")
        self.average_tweets_per_day: float = get_average_tweets_per_day(
            statuses_count=self.statuses_count, created_at=user.__getattribute__(
                "created_at"))
        self.account_age_days: int = get_account_age_in_days(
            created_at=user.__getattribute__("created_at"))


class UserAccountDoc(UserProfile):
    def __init__(self, user: User):
        super().__init__(user=user)
        self.name: str = user.__getattribute__("name")
        self.url: str = user.__getattribute__("url")
        try:
            self.profile_banner_url: str = user.__getattribute__("profile_banner_url")
        except Exception as e:
            self.profile_banner_url: str = \
                "https://i.picsum.photos/id/1021/2048/1206.jpg?hmac=fqT2NWHx783Pily1V_39ug_GFH1A4GlbmOMu8NWB3Ts"
        self.profile_url: str = "https://twitter.com/" + self.screen_name.replace("@", "")
        self.listed_count: str = user.__getattribute__("listed_count")
        self.botometer_analysis: dict = {}
        self.uuid: str = self.get_128_uuid(data_str=str(self.id))


class StatusDoc(TwitterDocument):
    def __init__(self, status: Status):
        self.created_at: str = preprocess_date(status.__getattribute__(
            "created_at"))
        self.id: int = status.__getattribute__("id")
        self.hashtags: list = status.__getattribute__("entities").get("hashtags", [])
        self.user_mentions: list = status.__getattribute__("entities").get("user_mentions", [])
        self.urls: list = status.__getattribute__("entities").get("urls", [])
        self.media: list = status.__getattribute__("entities").get("media", [])
        self.is_quote_status: bool = status.__getattribute__("is_quote_status")
        self.quote_count: int = status.__getattribute__("quote_count") if\
            hasattr(status, 'quote_count') else 0
        self.text: str = self.get_text(status=status)
        self.retweet_count: int = status.__getattribute__("retweet_count")
        self.retweeted: bool = status.__getattribute__("retweeted")
        self.user_id: int = status.__getattribute__("user").__getattribute__("id")
        self.possibly_sensitive: bool = status.__getattribute__("possibly_sensitive") if\
            hasattr(status, 'possibly_sensitive') else False
        self.favorite_count: int = status.__getattribute__("favorite_count")
        self.favorited: bool = status.__getattribute__("favorited")
        self.lang: str = detect(self.text)
        self.url: str = f"https://twitter.com/user/status/{status.__getattribute__('id')}"
        self.sentiment_analysis: dict = {}
        self.source: str = status.__getattribute__("source")
        self.coordinates: dict = status.__getattribute__("coordinates") if\
            hasattr(status, 'coordinates') else {}
        self.place: dict = self.get_place(status=status)
        self.reply_count: int = status.__getattribute__("reply_count") if\
            hasattr(status, 'quote_count') else 0
        self.uuid: str = self.get_128_uuid(data_str=str(self.id))
        self.user_uuid: str = self.get_128_uuid(data_str=str(self.user_id))

    @staticmethod
    def get_place(status: Status) -> dict:
        place: dict = {"coordinates": [], "country": "N/A",
                       "country_code": "N/A", "full_name": "N/A"}
        try:
            if status.__getattribute__("place") is not None:
                status_place: dict = status.__getattribute__("place").__dict__
                coordinates: list = status_place.get("bounding_box").__dict__.get(
                    "coordinates", [])
                country: str = status_place.get("country", "N/A")
                country_code: str = status_place.get("country_code", "N/A")
                full_name: str = status_place.get("full_name", "N/A")
                place: dict = {"coordinates": coordinates, "country": country,
                               "country_code": country_code, "full_name": full_name}
        except Exception as e:
            pass
        return place

    @staticmethod
    def get_text(status: Status) -> str:
        if hasattr(status, "full_text"):  # Check if Retweet
            try:
                text: str = status.__getattribute__("full_text")
            except AttributeError:
                text: str = status.__getattribute__("text")
        else:
            try:
                text: str = status.__getattribute__("extended_tweet").__getattribute__("full_text")
            except AttributeError:
                text: str = status.__getattribute__("text")
        return text


class TwitterDataOutput(object):
    def __init__(self, status: dict, user: dict):
        self.status: dict = status
        self.user: dict = user


class StatusTwintDoc(TwitterDocument):
    def __init__(self, status: tweet):
        self.created_at: str = preprocess_date(
            self.get_datetime(status=status))
        self.id: int = status.__getattribute__("id")
        self.hashtags: list = status.__getattribute__("hashtags")
        self.user_mentions: list = status.__getattribute__("mentions")
        self.urls: list = status.__getattribute__("urls")
        self.media: list = self.get_media(status=status)
        self.is_quote_status: bool = False
        self.quote_count: int = 0
        self.text: str = status.__getattribute__("tweet")
        self.retweet_count: int = status.__getattribute__("retweets_count")
        self.retweeted: bool = status.__getattribute__("retweeted") if \
            hasattr(status, 'retweeted') else False
        self.user_id: int = status.__getattribute__("user_id")
        self.possibly_sensitive: bool = status.__getattribute__("possibly_sensitive") if \
            hasattr(status, 'possibly_sensitive') else False
        self.favorite_count: int = status.__getattribute__("likes_count")
        self.favorited: bool = status.__getattribute__("favorited") if \
            hasattr(status, 'favorited') else False
        self.lang: str = detect(self.text)
        self.url: str = status.__getattribute__('link')
        self.sentiment_analysis: dict = {}
        self.source: str = status.__getattribute__("source")
        self.reply_count: int = status.__getattribute__("replies_count")

    @staticmethod
    def get_datetime(status: tweet):
        date_time_str = f"{status.__getattribute__('datestamp')} {status.__getattribute__('timestamp')}"
        date_time_obj: datetime = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
        return date_time_obj

    @staticmethod
    def get_media(status: tweet) -> list:
        media: list = status.__getattribute__("photos")
        return media


class UserTwintDoc(TwitterDocument):
    def __init__(self, user: TwintUser):
        self.created_at: str = preprocess_date(user.__getattribute__(
            "created_at"))
        self.default_profile: bool = user.__getattribute__(
            "default_profile")
        self.default_profile_image: bool = user.__getattribute__(
            "default_profile_image")
        self.description: str = user.__getattribute__(
            "bio")
        self.favourites_count: int = user.__getattribute__(
            "likes")
        self.followers_count: int = user.__getattribute__(
            "followers")
        self.friends_count: int = user.__getattribute__(
            "following")
        self.geo_enabled: bool = user.__getattribute__(
            "geo_enabled")
        self.id: int = user.__getattribute__("id")
        self.lang: str = self.get_language(text=user.__getattribute__(
            "lang")) if user.__getattribute__("lang") is not None else \
            self.get_language(text=user.__getattribute__("bio"))
        self.location: str = "N/A" if user.__getattribute__(
            "location") is None or user.__getattribute__(
            "location") == "" else user.__getattribute__("location")
        self.profile_background_image_url: str = user.__getattribute__(
            "background_image")
        self.profile_image_url: str = user.__getattribute__(
            "avatar")
        self.screen_name: str = user.__getattribute__(
            "username")
        self.statuses_count: int = user.__getattribute__(
            "tweets")
        self.verified: bool = user.__getattribute__(
            "is_verified")
        self.average_tweets_per_day: float = get_average_tweets_per_day(
            statuses_count=self.statuses_count, created_at=user.__getattribute__(
                "created_at"))
        self.account_age_days: int = get_account_age_in_days(
            created_at=user.__getattribute__("created_at"))
        self.name: str = user.__getattribute__("name")
        self.url: str = user.__getattribute__("url")
        try:
            self.profile_banner_url: str = user.__getattribute__("profile_banner_url")
        except Exception as e:
            self.profile_banner_url: str = \
                "https://i.picsum.photos/id/1021/2048/1206.jpg?hmac=fqT2NWHx783Pily1V_39ug_GFH1A4GlbmOMu8NWB3Ts"
        self.profile_url: str = "https://twitter.com/" + self.screen_name.replace("@", "")
        self.listed_count: int = user.__getattribute__("media_count ")
        self.botometer_analysis: dict = {}
