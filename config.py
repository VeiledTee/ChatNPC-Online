import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Config:
    def __init__(self, config_file):
        self._load_config(config_file)

    def _load_config(self, config_file):
        with open(config_file, "r") as f:
            config_data = json.load(f)

        # Load configuration parameters
        self.DATASET = config_data["DATASET"]
        self.INPUT_SIZE = config_data["INPUT_SIZE"]
        self.SEQUENCE_LENGTH = config_data["SEQUENCE_LENGTH"]
        self.HIDDEN_SIZE = config_data["HIDDEN_SIZE"]
        self.NUM_LAYERS = config_data["NUM_LAYERS"]
        self.OUTPUT_SIZE = config_data["OUTPUT_SIZE"]
        self.EPOCHS = config_data["EPOCHS"]
        self.LEARNING_RATE = config_data["LEARNING_RATE"]
        self.CHKPT_INTERVAL = config_data["CHKPT_INTERVAL"]
        self.TESTSET = config_data["TESTSET"]
        self.DATE_FORMAT = config_data["DATE_FORMAT"]
        self.BATCH_SIZE = (
            config_data["BATCH_SIZE_TRAIN"]
            if self.DATASET == "train"
            else config_data["BATCH_SIZE_OTHER"]
        )
        self.MODEL_NAME = config_data["MODEL_NAME"]

        # Set pytorch device
        if torch.cuda.is_available():
            self.DEVICE = torch.device("cuda")
        else:
            self.DEVICE = torch.device("cpu")

        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.MODEL = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        ).to(self.DEVICE)
