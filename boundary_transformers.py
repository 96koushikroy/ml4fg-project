import re
import torch
import torch.nn as nn
import torch.utils.data

from transformers import BertConfig, BertForMaskedLM


class Anchor_BERT_Model(nn.Module):
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

        configuration = BertConfig(max_position_embeddings=pos_enc_size,
                                      num_hidden_layers=12,
                                      num_labels=2)
        self.transformer = BertForMaskedLM(configuration)
        
    def forward(self, x):
        x = x.unsqueeze(1) #[*, 1, 4k, 5]
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition(x)
        logits = self.transformer(input_ids=x).logits
        predicted_class_id = logits.argmax(dim=1)
        return predicted_class_id.view(batch_size, -1)


class Anchor_BERTXL_Model(nn.Module):
    def __init__(self, pretrained_name=None, freeze_layers=0):
        super().__init__()
        self.seq_len = 4000
        self.freeze_layers = freeze_layers
        if pretrained_name:
            self.transformer = BertForMaskedLM.from_pretrained(pretrained_name)
        else:
            config = BertConfig(max_position_embeddings=512,
                                num_hidden_layers=12)
            self.transformer = BertForMaskedLM(config)

        # Freeze first few layers of the transformer
        for name, param in self.transformer.named_parameters():
            layer_num = re.search('\d+', name)
            if layer_num is not None and layer_num < self.freeze_layers:
                param.requires_grad = False

        self.num_chunks = self.seq_len // self.transformer.config['max_position_embeddings']

        self.head = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            nn.LeakyReLU(0.2),
            nn.Linear(self.seq_len, 1),
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_chunks = x.split(self.num_chunks, dim=1)

        output_chunks = []
        for chunk in x_chunks:
            logits = self.transformer(input_ids=chunk).logits
            output_chunks.append(logits)
        
        output = torch.cat(output_chunks, dim=1)

        pred = self.head(output)
        return pred.view(batch_size, -1)
