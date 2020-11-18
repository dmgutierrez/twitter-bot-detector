import coloredlogs, logging
import warnings, os
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)


# List of parameters to be considered
parent_dir: str = os.getenv("PARENT_DIR") if "PARENT_DIR" in os.environ else ""
result_params_dir: str = "pretrained_models"
model_directory: str = os.path.join(parent_dir, result_params_dir)
model_name: str = "BOT-NET-RoBERTa-transformer.h5"
history_directory: str = model_directory
scaler_filename: str = os.path.join(parent_dir, result_params_dir, "scaler.pkl")
embedding_layer: str = "embedding"

link_information: list = ["",
                          "",
                          ""]
join_attr: str = ". "
default_value: str = "unknown"
output_mapping: dict = {"human": 0, "bot": 1}
popularity_metric: str = "popularity"
id_col: str = "id"
boolean_cols: list = ["default_profile", "default_profile_image",
                      "geo_enabled", "verified"]
drop_num_cols: list = [id_col]
cat_cols: list = ["description", "screen_name", "lang"]

document_col: str = "document"
target_col: str = "account_type"
doc_emb_col: str = "doc_embedding"
uuid_col: str = "uuid"