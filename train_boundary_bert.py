import torch
from pathlib import Path
from boundary_transformers import Anchor_BERT_Model
from train_boundary import train_model

torch.manual_seed(2)


DATASET_ROOT = Path("D:\Datasets\Chromatin Loops")
DEEPMILO_ROOT = DATASET_ROOT / "deepmilo_data"
BOUNDARY_ROOT = DEEPMILO_ROOT / "boundary"

train_boundary_templ = str(BOUNDARY_ROOT / "data_boundary_train" / "data_boundary_traintest_{}.mat")
train_label_templ = str(BOUNDARY_ROOT / "label_boundary_train" / "label_boundary_traintest_{}.mat")

val_boundary_templ = str(BOUNDARY_ROOT / "data_boundary_val" / "data_boundary_valtest_{}.mat")
val_label_templ = str(BOUNDARY_ROOT / "label_boundary_val" / "label_boundary_valtest_{}.mat")

test_boundary_templ = str(BOUNDARY_ROOT / "data_boundary_test" / "data_boundary_testtest_{}.mat")
test_label_templ = str(BOUNDARY_ROOT / "label_boundary_test" / "label_boundary_testtest_{}.mat")

TRAIN_DATASET_LENGTH = 150104
VAL_DATASET_LENGTH  = 5920
TEST_DATASET_LENGTH = 23744
DATASET_LENGTHS = (TRAIN_DATASET_LENGTH, VAL_DATASET_LENGTH, TEST_DATASET_LENGTH)

POS_ENC_SIZE = 1024

config = {
    "name": "anchor_model_checkpoint_bert.pt",
    "batch_size": 2,
    "epochs": 15,
    "patience": 4,
    "verbose": True,
    "data_config" : {
        "rnn_len":  2,
        "argmax": True
    }
}


if __name__ == "__main__":
    model = Anchor_BERT_Model(pos_enc_size=POS_ENC_SIZE)
    my_cnn, train_accs, val_accs = train_model(model, (train_boundary_templ, train_label_templ), (val_boundary_templ, val_label_templ), DATASET_LENGTHS, config)

    torch.save({
        'train_accs': train_accs,
        'val_accs': val_accs,
        'state_dict': my_cnn.state_dict()
    }, "deepmilo_boundary_cnn_lstm.pt")