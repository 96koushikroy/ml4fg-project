import torch
from pathlib import Path
from boundary_transformers import Anchor_Enformer_Model
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

# TRAIN_DATASET_LENGTH = 150104
TRAIN_DATASET_LENGTH = 50000
VAL_DATASET_LENGTH  = 5920
TEST_DATASET_LENGTH = 23744

DATASET_LENGTHS = (TRAIN_DATASET_LENGTH, VAL_DATASET_LENGTH, TEST_DATASET_LENGTH)

config = {
    "name": "anchor_model_checkpoint_enformer_2.pt",
    "batch_size": 4,
    "lr": 1e-5,
    "epochs": 15,
    "patience": 4,
    "verbose": True,
    "data_config" : {
        "rnn_len":  2,
        "argmax": True
    },
    "train": {
        "cast": 'long',
        "use_rnn": False
    }
}


if __name__ == "__main__":
    model = Anchor_Enformer_Model(dim=192, depth=8, target_length=32, num_downsamples=3)
    # model.load_state_dict(torch.load('anchor_model_checkpoint_enformer.pt')['model_state_dict'])
    model, train_accs, val_accs = train_model(model, (train_boundary, train_label), (val_boundary, val_label), DATASET_LENGTHS, config)

    torch.save({
        'train_accs': train_accs,
        'val_accs': val_accs,
        'model_state_dict': model.state_dict(),
    }, "deepmilo_boundary_enformer_2.pt")
