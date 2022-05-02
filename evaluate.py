import math
import sklearn
import torch
from pathlib import Path
from transformers import AutoTokenizer

from boundary_models import Anchor_CNN_LSTM
from boundary_transformers import Anchor_Enformer_Model, Anchor_BERTXL_Model
from boundary_dataset import AnchorDataset, AnchorCollate
from train_boundary import run_one_epoch

DATASET_ROOT = Path("./data")#Path("D:\Datasets\Chromatin Loops")
DEEPMILO_DATA = DATASET_ROOT / "deepmilo_data"
TEST_TYPES = DEEPMILO_DATA / "negative_types" / "negative_types_processed"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_dataset_lengths = {
    1: 23686,
    2: 79026,
    3: 124930
}

data_config = {
        "rnn_len":  2,
        "argmax": True,
        # "decode_bp": True,
        # "kmer": 6
    }

test_config = {
        "cast": 'long',
        "use_rnn": False,
    }

# tokenizer = AutoTokenizer.from_pretrained("armheb/DNA_bert_6")
tokenizer = None
collate = AnchorCollate(tokenizer) if tokenizer is not None else None

batch_size = 8
name = "enformer_pretrained"

def evaluate_model(name, dataloader, dataset, config):
    test_loss, test_acc, test_pr, test_rec = run_one_epoch(False, dataloader, model, None, device, math.ceil(len(dataset)/batch_size), 0, config)
    test_auprc = sklearn.metrics.auc(test_rec, test_pr)
    results = {
        'loss': test_loss,
        'acc': test_acc,
        'precision': test_pr,
        'recall': test_rec,
        'auprc': test_auprc
    }
    print(f"{name} AUPRC: {test_auprc}")
    torch.save(results, f"results_{name}.json")
    return results

if __name__ == "__main__":
    # model = Anchor_Enformer_Model(dim=192, depth=4, target_length=32, num_downsamples=3)
    # model.load_state_dict(torch.load('anchor_model_checkpoint_enformer.pt')['model_state_dict'])
    # model = Anchor_Enformer_Model(dim=384, depth=8, target_length=64, num_downsamples=3)
    # model.load_state_dict(torch.load('anchor_model_checkpoint_enformer_3.pt')['model_state_dict'])
    model = Anchor_Enformer_Model(dim=1536, target_length=16, pretrained=True, freeze_layers=9)
    model.load_state_dict(torch.load('anchor_model_checkpoint_enformer_pretrained.pt')['model_state_dict'])
    # model = Anchor_CNN_LSTM(use_cnn=True, use_lstm=True)
    # model.load_state_dict(torch.load('deepmilo_boundary_cnn_lstm.pt')['state_dict'])
    # model = Anchor_BERTXL_Model(pretrained_name="armheb/DNA_bert_6", freeze_layers=9)
    # model.load_state_dict(torch.load('anchor_model_checkpoint_dnabert.pt')['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Evaluate on main test data
    test_boundary = str(DEEPMILO_DATA / "anchor_processed" / "data_boundary_4k_testtest.mat")
    test_label = str(DEEPMILO_DATA / "anchor_processed" / "label_boundary_4k_testtest.mat")
    test_dataset = AnchorDataset(test_boundary, test_label, length=23744, **data_config)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers = 0, collate_fn=collate)
    results = evaluate_model(f"{name}_test", test_dataloader, test_dataset, test_config)

    # Evaluate on negative test sets
    test_type_files = dict()
    for i in range(1, 4):
        d = TEST_TYPES / f"data_boundary_4k_test_type{i}.mat"
        l = TEST_TYPES / f"label_boundary_4k_test_type{i}.mat"
        test_type_files[i] = (d, l)

    test_datasets = dict()
    test_dataloaders = dict()
    for i in range(1, 4):
        test_dataset = AnchorDataset(test_type_files[i][0], test_type_files[i][1], length=test_dataset_lengths[i], **data_config)
        test_datasets[i] = test_dataset
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers = 0, collate_fn=collate)
        test_dataloaders[i] = test_dataloader

    test_results = dict()
    for i in range(1, 4):
        results = evaluate_model(f"{name}_type_{i}", test_dataloaders[i], test_datasets[i], test_config)
        test_results[i] = results
