import torch
import torch.nn as nn
import torch.utils.data

from transformers import RobertaConfig, RobertaForSequenceClassification


class Anchor_RoBERTa_Model(nn.Module):
    def __init__(self, conv_out=256, pos_enc_size=1024, dropout=0.2):
        super().__init__()
        self.seq_len = 4000
        
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels=1, out_channels=conv_out, kernel_size=20, padding='same'),
                        nn.BatchNorm1d(conv_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(2),
                        nn.Dropout2d(dropout),
                    )

        self.conv2 = nn.Sequential(
                        nn.Conv1d(in_channels=conv_out, out_channels=conv_out, kernel_size=10, padding='same'),
                        nn.BatchNorm1d(conv_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(2),
                        nn.Dropout2d(dropout),
                    )
        
        self.transition = nn.Sequential(
            nn.Linear(conv_out, 1),
            nn.LeakyReLU(0.2)
        )

        configuration = RobertaConfig(max_position_embeddings=pos_enc_size,
                                      num_hidden_layers=12,
                                      num_labels=2)
        self.transformer = RobertaForSequenceClassification(configuration)
        
    def forward(self, x):
        x = x.unsqueeze(1) #[*, 1, 4k, 5]
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition(x)
        logits = self.transformer(input_ids=x).logits
        predicted_class_id = logits.argmax().item()
        return predicted_class_id.view(batch_size, -1)
