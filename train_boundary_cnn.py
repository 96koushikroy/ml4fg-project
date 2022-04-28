import torch
from pathlib import Path
from boundary_models import Anchor_CNN_LSTM
from train_boundary import train_model

torch.manual_seed(2)


DATASET_ROOT = Path("D:\Datasets\Chromatin Loops")
DEEPMILO_ROOT = DATASET_ROOT / "deepmilo_data"
BOUNDARY_ROOT = DEEPMILO_ROOT / "boundary"

train_boundary = str(DEEPMILO_ROOT / "anchor_processed" / "data_boundary_4k_traintest.mat")
train_label = str(DEEPMILO_ROOT / "anchor_processed" / "label_boundary_4k_traintest.mat")

val_boundary = str(DEEPMILO_ROOT / "anchor_processed" / "data_boundary_4k_valtest.mat")
val_label = str(DEEPMILO_ROOT / "anchor_processed" / "label_boundary_4k_valtest.mat")

test_boundary = str(DEEPMILO_ROOT / "anchor_processed" / "data_boundary_4k_testtest.mat")
test_label = str(DEEPMILO_ROOT / "anchor_processed" / "label_boundary_4k_testtest.mat")

TRAIN_DATASET_LENGTH = 150104
VAL_DATASET_LENGTH  = 5920
TEST_DATASET_LENGTH = 23744
DATASET_LENGTHS = (TRAIN_DATASET_LENGTH, VAL_DATASET_LENGTH, TEST_DATASET_LENGTH)

config = {
    "name": "anchor_model_checkpoint_cnn.pt",
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 15,
    "patience": 4,
    "verbose": True,
    "data_config" : {
        "rnn_len":  2,
        "argmax": False
    }
}


if __name__ == "__main__":
    my_cnn = Anchor_CNN_LSTM(use_cnn=True, use_lstm=False)
    my_cnn, train_accs, val_accs = train_model(my_cnn, (train_boundary, train_label), (val_boundary, val_label), DATASET_LENGTHS, config)
