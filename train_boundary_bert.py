import torch
from pathlib import Path
from boundary_transformers import Anchor_BERT_Model, Anchor_BERT_Model_2
from train_boundary import train_model

torch.manual_seed(2)


DATASET_ROOT = Path("D:\Datasets\Chromatin Loops")
DEEPMILO_ROOT = DATASET_ROOT / "deepmilo_data"
BOUNDARY_ROOT = DEEPMILO_ROOT / "boundary"

# train_boundary_templ = str(BOUNDARY_ROOT / "data_boundary_train" / "data_boundary_traintest_{}.mat")
# train_label_templ = str(BOUNDARY_ROOT / "label_boundary_train" / "label_boundary_traintest_{}.mat")

# val_boundary_templ = str(BOUNDARY_ROOT / "data_boundary_val" / "data_boundary_valtest_{}.mat")
# val_label_templ = str(BOUNDARY_ROOT / "label_boundary_val" / "label_boundary_valtest_{}.mat")

# test_boundary_templ = str(BOUNDARY_ROOT / "data_boundary_test" / "data_boundary_testtest_{}.mat")
# test_label_templ = str(BOUNDARY_ROOT / "label_boundary_test" / "label_boundary_testtest_{}.mat")

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

POS_ENC_SIZE = 512

config = {
    "name": "anchor_model_checkpoint_bert.pt",
    "batch_size": 4,
    "lr": 1e-3,
    "epochs": 15,
    "patience": 4,
    "verbose": True,
    "data_config" : {
        "rnn_len":  2,
        "argmax": False
    },
    "train": {
        "cast": 'float',
        "use_rnn": False
    }
}


if __name__ == "__main__":
    # model = Anchor_BERT_Model(pos_enc_size=POS_ENC_SIZE, hidden_size=256, hidden_layers=6)
    model = Anchor_BERT_Model_2(pos_enc_size=POS_ENC_SIZE, hidden_size=512, hidden_layers=2)
    
    model, train_accs, val_accs = train_model(model, (train_boundary, train_label), (val_boundary, val_label), DATASET_LENGTHS, config)

    torch.save({
        'train_accs': train_accs,
        'val_accs': val_accs,
        'model_state_dict': model.state_dict(),
    }, "deepmilo_boundary_bert.pt")
