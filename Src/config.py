import os 
from dotenv import load_dotenv
import tensorflow as tf 
import joblib
from .utils import model_reconstruct
from .constants import DOWNLOADED_IMAGES_PATHS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
tf.get_logger().setLevel('ERROR')
load_dotenv(override=True)
APP_NAME = os.getenv('APP_NAME')
API_SECRET_KEY = os.getenv('API_SECRET_KEY')
VERSION = os.getenv('VERSION')


os.makedirs(DOWNLOADED_IMAGES_PATHS, exist_ok=True)
#MODEL =tf.keras.models.load_model(os.path.join(os.getcwd(),'artifacts','model.keras'),compile=False)
IDX2LABEL = joblib.load(os.path.join(os.getcwd(),'artifacts','idx2label.joblib'))


WEIGHTS_PATH = os.path.join(os.getcwd(),'artifacts','model.weights.h5')
MODEL = model_reconstruct()
MODEL.load_weights(WEIGHTS_PATH)