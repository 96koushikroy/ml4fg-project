import re
import torch
import torch.nn as nn
import torch.utils.data
from einops.layers.torch import Rearrange
from enformer_pytorch import Enformer

from transformers import BertConfig, BertModel, BertForSequenceClassification
from boundary_models import Anchor_CNN_Stack

class Anchor_BERT_Model(nn.Module):
    def __init__(self, conv_out=64, pos_enc_size=1024, dropout=0.2, hidden_size=512, hidden_layers=12):
        super().__init__()
        self.seq_len = 4000
        
        self.conv1 = nn.Sequential(
                        nn.Conv1d(in_channels=5, out_channels=conv_out, kernel_size=19, padding='same'),
                        nn.BatchNorm1d(conv_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(2),
                        nn.Dropout(dropout),
                    )

        self.conv2 = nn.Sequential(
                        nn.Conv1d(in_channels=conv_out, out_channels=conv_out, kernel_size=19, padding='same'),
                        nn.BatchNorm1d(conv_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(2),
                        nn.Dropout(dropout),
                    )

        self.conv3 = nn.Sequential(
                        nn.Conv1d(in_channels=conv_out, out_channels=conv_out, kernel_size=19, padding='same'),
                        nn.BatchNorm1d(conv_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(2),
                        nn.Dropout(dropout),
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

class Anchor_BERT_Model_2(nn.Module):
    def __init__(self, pos_enc_size=1024, dropout=0.2, hidden_size=512, hidden_layers=12):
        super().__init__()
        self.seq_len = 4000
        
        self.conv_block = Anchor_CNN_Stack(layer_dims=[[1, 256], [256, 512]], kernel_sizes=[(17, 5), (5, 1)], dropout=dropout)
        self.conv_block_2 = Anchor_CNN_Stack(layer_dims=[[512, 256], [256, 16]], kernel_sizes=[(5, 1), (5, 1)], dropout=dropout)
        
        configuration = BertConfig(max_position_embeddings=pos_enc_size,
                                   hidden_size=hidden_size,
                                   num_attention_heads=hidden_size//64,
                                   num_hidden_layers=hidden_layers,
                                   output_hidden_states=True)
        self.transformer =  BertModel(configuration)

    
        self.classifier = nn.Sequential(
                            nn.Linear(112, 64),
                            nn.BatchNorm1d(64),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.2),
                            
                            nn.Linear(64, 1),
                            nn.Sigmoid()
                        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        # x = x.view(batch_size, x.shape[2], -1)
        x = self.conv_block(x)
        # x = x.view(batch_size, x.shape[2], -1)
        print(x.shape)
        x = self.transformer(inputs_embeds=x).last_hidden_state
        # x = x.view(batch_size, x.shape[2], -1)
        x = self.conv_block_2(x)
        # x = x.view(batch_size, -1)

        predicted_class_id = self.classifier(x)
        return predicted_class_id.view(batch_size, -1)


class Anchor_BERTXL_Model(nn.Module):
    def __init__(self, pretrained_name=None, num_layers=0, freeze_layers=0):
        super().__init__()
        self.seq_len = 4096
        self.num_layers = num_layers
        self.freeze_layers = freeze_layers
        if pretrained_name is not None:
            self.transformer = BertModel.from_pretrained(pretrained_name, output_hidden_states=True) #num_hidden_layers=self.num_layers
        else:
            config = BertConfig(max_position_embeddings=512,
                                num_hidden_layers=12,
                                num_labels=2)
            self.transformer = BertModel(config)

        # Freeze first few layers of the transformer
        for name, param in self.transformer.named_parameters():
            layer_num = re.search('\d+', name)
            layer_num = int(layer_num[0]) if layer_num is not None else 1e4
            if 'bert.embeddings' in name or layer_num <= self.freeze_layers:
                param.requires_grad = False


        self.num_chunks = self.seq_len // self.transformer.config.max_position_embeddings

        self.transition = nn.Sequential(
            Rearrange('b l c -> b c l'),
            nn.MaxPool1d(4),
            Rearrange('b c l -> b l c'),
            nn.Linear(self.transformer.config.hidden_size, 16),
            Rearrange('b l c -> b c l'),
            nn.MaxPool1d(4),
            Rearrange('b c l -> b l c'),
        )
        
        # self.transformer.config.max_position_embeddings: 512
        self.head = nn.Sequential(
            nn.Linear(3968, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x['input_ids'].shape[0]
        input_id_chunks = x['input_ids'].tensor_split(self.num_chunks, dim=1)
        token_type_id_chunks = x['token_type_ids'].tensor_split(self.num_chunks, dim=1)
        attention_mask_chunks = x['attention_mask'].tensor_split(self.num_chunks, dim=1)

        # seps = 3*torch.ones(batch_size, 1, dtype=torch.long, device=torch.device('cuda'))

        output_chunks = []
        for chunk in range(self.num_chunks):
            # Hack to add a [SEP] token to the end of each input
            # input_ids = torch.cat((input_id_chunks[chunk], seps), 1)
            # token_type_ids = torch.cat((token_type_id_chunks[chunk], seps), 1)
            # attention_masks = torch.cat((attention_mask_chunks[chunk], seps), 1)
            input_ids, token_type_ids, attention_masks = input_id_chunks[chunk], token_type_id_chunks[chunk], attention_mask_chunks[chunk]
            transformer_output = self.transformer(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_masks)
            trans_out = self.transition(transformer_output.hidden_states[-1]).squeeze()
            # print(len(transformer_output.hidden_states))
            output_chunks.append(trans_out)
        
        output = torch.cat(output_chunks, dim=1).view(batch_size, -1)
        pred = self.head(output)
        return pred.view(batch_size, -1)

# Enformer Model 
# The Dimensionality reduction is from 4,000 --> 512 (4x reduction)

class Anchor_Enformer_Model(nn.Module):
    def __init__(self, dim=192, depth=4, target_length=32, num_downsamples=0, pretrained=False, freeze_layers=0):
        super().__init__()
        self.freeze_layers = freeze_layers
        if pretrained:
            self.enformer = Enformer.from_pretrained('EleutherAI/enformer-preview', target_length = target_length, use_checkpointing = True)
            for name, param in self.enformer.named_parameters():
                if name.startswith('stem.'):
                    param.requires_grad = False
                elif name.startswith('conv_tower.'):
                    param.requires_grad = False
                elif name.startswith('transformer.'):
                    layer_num = int(re.search('transformer\.(\d+)', name).group(1))
                    if layer_num <= self.freeze_layers:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

        else:
            self.enformer = Enformer.from_hparams(
                dim = dim,
                depth = depth,
                heads = 8,
                # output_heads = dict(),
                output_heads = dict(human = 5313, mouse = 1643),
                target_length = target_length,
                num_downsamples=num_downsamples,
                dim_divisible_by = 8
            )

        self.transition = torch.nn.Linear(dim*2, 16)

        self.classifier = nn.Sequential(
                            nn.Linear(target_length*16, 512),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.2),
            
                            nn.Linear(512, 256),
                            nn.BatchNorm1d(256),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.2),
                            
                            nn.Linear(256, 1),
                            nn.Sigmoid()
                        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.enformer(x, return_only_embeddings=True)
        x = self.transition(x)
        x = x.view(batch_size, -1)
        out = self.classifier(x)
        return out
