import torch
from pathlib import Path
from boundary_transformers import Anchor_BERTXL_Model
from train_boundary import train_model
from transformers import AutoTokenizer

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

config = {
    "name": "anchor_model_checkpoint_bert.pt",
    "batch_size": 8,
    "lr": 1e-4,
    "epochs": 15,
    "patience": 4,
    "verbose": True,
    "tokenizer": AutoTokenizer.from_pretrained("../DNA_bert_6"),
    "data_config" : {
        "rnn_len":  2,
        "argmax": True
    }
}

POS_ENC_SIZE = 512

if __name__ == "__main__":
    model = Anchor_BERTXL_Model(pretrained_name="../DNA_bert_6", num_layers=6, freeze_layers=6)
    model, train_accs, val_accs = train_model(model, (train_boundary_templ, train_label_templ), (val_boundary_templ, val_label_templ), DATASET_LENGTHS, config)

    torch.save({
        'train_accs': train_accs,
        'val_accs': val_accs,
        'model_state_dict': model.state_dict(),
    }, "deepmilo_boundary_dnabert.pt")
