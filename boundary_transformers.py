import re
import torch
import torch.nn as nn
import torch.utils.data

from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, BertForPreTraining, BertForSequenceClassification


class Anchor_BERT_Model(nn.Module):
    def __init__(self, conv_out=64, pos_enc_size=1024, dropout=0.2, hidden_size=512, hidden_layers=12):
        super().__init__()
        self.seq_len = 4000
        
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels=5, out_channels=conv_out, kernel_size=19, padding='same'),
                        nn.BatchNorm1d(conv_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(2),
                        nn.Dropout2d(dropout),
                    )

        self.conv2 = nn.Sequential(
                        nn.Conv1d(in_channels=conv_out, out_channels=conv_out, kernel_size=9, padding='same'),
                        nn.BatchNorm1d(conv_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(2),
                        nn.Dropout2d(dropout),
                    )

        self.conv3 = nn.Sequential(
                        nn.Conv1d(in_channels=conv_out, out_channels=conv_out, kernel_size=9, padding='same'),
                        nn.BatchNorm1d(conv_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(2),
                        nn.Dropout2d(dropout),
                    )

        self.conv_block = nn.Sequential(self.conv1, self.conv2, self.conv3)
        
        self.transition = nn.Sequential(
            nn.Linear(conv_out, hidden_size)
        )

        configuration = BertConfig(max_position_embeddings=pos_enc_size,
                                   hidden_size=hidden_size,
                                   num_attention_heads=hidden_size//64,
                                   num_hidden_layers=hidden_layers,
                                   num_labels=2)
        self.transformer =  BertForSequenceClassification(configuration)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, x.shape[2], -1)
        x = self.conv_block(x)
        x = x.view(batch_size, x.shape[2], -1)
        x = self.transition(x)
        logits = self.transformer(inputs_embeds=x).logits
        predicted_class_id = logits.softmax(dim=1)[:, 1]
        return predicted_class_id.view(batch_size, -1)


class Anchor_BERTXL_Model(nn.Module):
    def __init__(self, pretrained_name=None, freeze_layers=0):
        super().__init__()
        self.seq_len = 4096
        self.freeze_layers = freeze_layers
        if pretrained_name:
            self.transformer = BertForPreTraining.from_pretrained(pretrained_name, output_hidden_states=True)
        else:
            config = BertConfig(max_position_embeddings=512,
                                num_hidden_layers=12,
                                num_labels=2)
            self.transformer = BertForPreTraining(config)

        # Freeze first few layers of the transformer
        for name, param in self.transformer.named_parameters():
            layer_num = re.search('\d+', name)
            layer_num = int(layer_num[0]) if layer_num is not None else 1e4
            if 'bert.embeddings' in name or layer_num <= self.freeze_layers:
                param.requires_grad = False


        self.num_chunks = self.seq_len // self.transformer.config.max_position_embeddings

        self.transition = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 1)
        )
        
        # self.transformer.config.max_position_embeddings: 512
        self.head = nn.Sequential(
            nn.Linear(self.transformer.config.max_position_embeddings, self.transformer.config.max_position_embeddings),
            nn.LeakyReLU(0.2),
            nn.Linear(self.transformer.config.max_position_embeddings, 2),
        )
        
    def forward(self, x):
        batch_size = x['input_ids'].shape[0]
        input_id_chunks = x['input_ids'].tensor_split(self.num_chunks, dim=1)
        token_type_id_chunks = x['token_type_ids'].tensor_split(self.num_chunks, dim=1)
        attention_mask_chunks = x['attention_mask'].tensor_split(self.num_chunks, dim=1)

        output_chunks = []
        for chunk in range(self.num_chunks):
            transformer_output = self.transformer(input_ids=input_id_chunks[chunk],
                                      token_type_ids=token_type_id_chunks[chunk],
                                      attention_mask=attention_mask_chunks[chunk])
            trans_out = self.transition(transformer_output.hidden_states[-1]).squeeze()
            # print(len(transformer_output.hidden_states))
            # predicted_class_prob = logits.softmax(dim=1)[:, 1]
            output_chunks.append(trans_out)
        
        output = torch.cat(output_chunks, dim=1) # [*, 512]
        pred = self.head(output).softmax(dim=1)[:, 1]
        return pred.view(batch_size, -1)
