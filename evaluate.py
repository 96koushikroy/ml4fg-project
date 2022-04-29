import math
import sklearn
import torch
from pathlib import Path

from boundary_transformers import Anchor_Enformer_Model
from boundary_dataset import AnchorDataset, AnchorCollate
from train_boundary import run_one_epoch

DATASET_ROOT = Path("D:\Datasets\Chromatin Loops")
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
        "argmax": True
    }

test_config = {
        "cast": 'long',
        "use_rnn": False
    }

batch_size = 2

if __name__ == "__main__":
    # model = Anchor_Enformer_Model(dim=192, depth=4, target_length=32, num_downsamples=3)
    model = Anchor_Enformer_Model(dim=384, depth=8, target_length=64, num_downsamples=3)
    model.load_state_dict(torch.load('anchor_model_checkpoint_enformer_3.pt')['model_state_dict'])
    model = model.to(device)
    model.eval()

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
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers = 0)
        test_dataloaders[i] = test_dataloader

    test_results = dict()
    for i in range(1, 4):
        test_loss, test_acc, test_pr, test_rec = run_one_epoch(False, test_dataloaders[i], model, None, device, math.ceil(len(test_datasets[i])/batch_size), i-1, test_config)
        test_auprc = sklearn.metrics.auc(test_rec, test_pr)
        results = {
            'loss': test_loss,
            'acc': test_acc,
            'precision': test_pr,
            'recall': test_rec,
            'auprc': test_auprc
        }
        print(f"Type {i} AUPRC: {test_auprc}")
        torch.save(results, f"results_type{i}.json")
        test_results[i] = results
