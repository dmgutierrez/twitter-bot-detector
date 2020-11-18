from collections import defaultdict

class GeneralAPIResponse(object):
    def __init__(self, status_code: int, message: str, data: dict):
        self.status_code: int = status_code
        self.message: str = message
        self.data: dict = data


class StreamingServiceOutput(object):
    def __init__(self, status_code: int, message: str, data: dict):
        self.status_code: int = status_code
        self.message: str = message
        self.data: dict = data


class StreamingProcessOutput(object):
    def __init__(self, status_code: int, message: str, data: dict):
        self.status_code: int = status_code
        self.message: str = message
        self.data: dict = data


class FlairSentOutput(object):
    def __init__(self, analysed: bool = False, label: str = "N/A",
                 confidence: float = 0.0, translated: bool = True):
        self.analysed: bool = analysed
        self.label: str = label
        self.confidence: float = confidence
        self.translated: bool = translated


class TextBlobSentOutput(object):
    def __init__(self, analysed: bool = False, subjectivity: float = 0.0,
                 polarity: float = 0.0, translated: bool = True):
        self.analysed: bool = analysed
        self.subjectivity: float = subjectivity
        self.polarity: float = polarity
        self.translated: bool = translated


class NLTKSentOutput(object):
    def __init__(self, analysed: bool = False, neutral_prob: float = 0.0,
                 negative_prob: float = 0.0, positive_prob: float = 0.0,
                 compound_prob: float = 0.0, translated: bool = True):
        self.analysed: bool = analysed
        self.neutral_prob: float = neutral_prob
        self.negative_prob: float = negative_prob
        self.positive_prob: float = positive_prob
        self.compound_prob: float = compound_prob
        self.translated: bool = translated


class SentimentAnalysisOutput(object):
    def __init__(self, flair_sent: FlairSentOutput = None, textblob_sent: TextBlobSentOutput = None,
                 nltk_sent: NLTKSentOutput = None):
        self.flair_analysis: dict = flair_sent.__dict__ if flair_sent is not None else {}
        self.textblob_analysis: dict = textblob_sent.__dict__ if textblob_sent is not None else {}
        self.nltk_analysis: dict = nltk_sent.__dict__ if nltk_sent is not None else {}


class BotometerAnalysisOutput(object):
    def __init__(self, cap: dict = None, display_scores: dict = None, analysed: bool = False):
        self.analysed: bool = analysed
        self.english_score: float = cap.get("english", 0.0) if cap is not None else 0.0
        self.universal_score: float = cap.get("universal", 0.0) if cap is not None else 0.0
        self.english_display_scores: dict = display_scores.get("english", {}) if\
            display_scores is not None else {}
        self.universal_display_scores: dict = display_scores.get("universal", {}) if\
            display_scores is not None else {}