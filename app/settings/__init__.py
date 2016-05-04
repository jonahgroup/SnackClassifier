import os
import ConfigParser

from unipath import FSPath

PROJECT_DIR = FSPath(__file__).absolute().ancestor(3)
default_config_dir = os.sep.join((PROJECT_DIR, 'configuration'))

#
# Environment Settings
#
SETTINGS_FILE_NAME = os.sep.join((default_config_dir, 'environment.ini'))

if not os.path.isfile(SETTINGS_FILE_NAME):
    raise Exception("Environment settings file '%s' not found."
                    % SETTINGS_FILE_NAME)

config = ConfigParser.RawConfigParser()
config.read(SETTINGS_FILE_NAME)

PORT = int(config.get('system', 'PORT'))
IMAGE_FOLDER_PATH = config.get('classifier', 'IMAGE_FOLDER_PATH')
CLASSIFIER_TYPE = config.get('classifier', 'CLASSIFIER_TYPE')
CLASSIFIER_TRAIN_PATH = config.get('classifier', 'CLASSIFIER_TRAIN_PATH')
