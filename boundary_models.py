import torch
import torch.nn as nn
import torch.utils.data


class Anchor_LSTM_Model(nn.Module):
    def __init__(self, in_dim=5, hid_dim=128, out_dim=1):
        super().__init__()
        self.lstm1 = nn.LSTM(in_dim, hid_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hid_dim * 2, hid_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hid_dim * 2, out_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x).view(batch_size, -1)

    
class Anchor_CNN_Model(nn.Module):
    def __init__(self, layer1_out=256, layer2_out=512, dropout=0.2):
        super().__init__()
        self.seq_len = 4000
        
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=layer1_out, kernel_size=(17,5)),
                        nn.BatchNorm2d(layer1_out),
                        nn.LeakyReLU(0.2),
                        nn.Dropout2d(dropout),
                    )
        
        #dilated conv dilation=1
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=(5,1), dilation=1, padding='same'),
                        nn.BatchNorm2d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(kernel_size=(self.seq_len - 16, 1), stride=(self.seq_len - 16, 1)),
                        nn.Dropout2d(dropout),
                    )
        
        #dilated conv dilation=3
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=(5,1), dilation=3, padding='same'),
                        nn.BatchNorm2d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(kernel_size=(self.seq_len - 16, 1), stride=(self.seq_len - 16, 1)),
                        nn.Dropout2d(dropout),
                    )
        
        #dilated conv dilation=7
        self.layer2_3 = nn.Sequential(
                        nn.Conv2d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=(5,1), dilation=7, padding='same'),
                        nn.BatchNorm2d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(kernel_size=(self.seq_len - 16, 1), stride=(self.seq_len - 16, 1)),
                        nn.Dropout2d(dropout),
                    )
        
    def forward(self, x):
        x = x.unsqueeze(1) #[*, 1, 4k, 5]
        batch_size = x.shape[0]
        x = self.layer1(x)
        o1 = self.layer2_1(x).view(batch_size, -1)
        o2 = self.layer2_2(x).view(batch_size, -1)
        o3 = self.layer2_3(x).view(batch_size, -1)
        out = torch.cat((o1,o2,o3), -1)
        return out

class Anchor_CNN_Transformer(nn.Module):
    def __init__(self, layer1_out=256, layer2_out=512, dropout=0.2):
        super().__init__()
        self.seq_len = 4000
        self.stride = 4
        
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_channels=1, out_channels=layer1_out, kernel_size=(17,5)),
                        nn.BatchNorm2d(layer1_out),
                        nn.LeakyReLU(0.2),
                        nn.Dropout2d(dropout),
                    )
        
        #dilated conv dilation=1
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=(5,1), dilation=1, padding='same'),
                        nn.BatchNorm2d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(kernel_size=(self.stride, 1), stride=(self.stride, 1)),
                        nn.Dropout2d(dropout),
                    )
        
        #dilated conv dilation=3
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=(5,1), dilation=3, padding='same'),
                        nn.BatchNorm2d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(kernel_size=(self.stride, 1), stride=(self.stride, 1)),
                        nn.Dropout2d(dropout),
                    )
        
        #dilated conv dilation=7
        self.layer2_3 = nn.Sequential(
                        nn.Conv2d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=(5,1), dilation=7, padding='same'),
                        nn.BatchNorm2d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(kernel_size=(self.stride, 1), stride=(self.stride, 1)),
                        nn.Dropout2d(dropout),
                    )
        
    def forward(self, x):
        x = x.unsqueeze(1) #[*, 1, 4k, 5]
        x = self.layer1(x)
        o1 = self.layer2_1(x)
        o2 = self.layer2_2(x)
        o3 = self.layer2_3(x)
        out = torch.cat((o1,o2,o3), 1)
        return out

class Anchor_CNN_Stack(nn.Module):
    def __init__(self, layer_dims=[], kernel_sizes=[], dropout=0.2):
        super().__init__()
        self.seq_len = 4000
        self.stride = 16

        self.layers = nn.ModuleList()
        for dim, kernel in zip(layer_dims, kernel_sizes):
            layer = nn.Sequential(
                        nn.Conv2d(in_channels=dim[0], out_channels=dim[1], kernel_size=kernel, dilation=3, padding='same'),
                        nn.BatchNorm2d(dim[1]),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool2d(kernel_size=(self.stride, 1), stride=(self.stride, 1)),
                        nn.Dropout2d(dropout),
                    )
            self.layers.append(layer)
        
    def forward(self, x):
        x = x.unsqueeze(1) #[*, 1, 4k, 5]
        batch_size = x.shape[0]
        for layer in self.layers:
            x = layer(x)
        # out = x.view(batch_size, -1)
        out = x
        return out


class Anchor_CNN1D_Model(nn.Module):
    def __init__(self, layer1_out=256, layer2_out=512, dropout=0.2):
        super().__init__()
        self.seq_len = 4000
        
        self.layer1 = nn.Sequential(
                        nn.Conv1d(in_channels=5, out_channels=layer1_out, kernel_size=17),
                        nn.BatchNorm1d(layer1_out),
                        nn.LeakyReLU(0.2),
                        # nn.Dropout(dropout),
                    )
        
        #dilated conv dilation=1
        self.layer2_1 = nn.Sequential(
                        nn.Conv1d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=5, dilation=1, padding='same'),
                        nn.BatchNorm1d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(kernel_size=(self.seq_len - 16), stride=(self.seq_len - 16)),
                        # nn.Dropout(dropout),
                    )
        
        #dilated conv dilation=3
        self.layer2_2 = nn.Sequential(
                        nn.Conv1d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=5, dilation=3, padding='same'),
                        nn.BatchNorm1d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(kernel_size=(self.seq_len - 16), stride=(self.seq_len - 16)),
                        # nn.Dropout(dropout),
                    )
        
        #dilated conv dilation=7
        self.layer2_3 = nn.Sequential(
                        nn.Conv1d(in_channels=layer1_out, out_channels=layer2_out, kernel_size=5, dilation=7, padding='same'),
                        nn.BatchNorm1d(layer2_out),
                        nn.LeakyReLU(0.2),
                        nn.MaxPool1d(kernel_size=(self.seq_len - 16), stride=(self.seq_len - 16)),
                        # nn.Dropout(dropout),
                    )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, x.shape[2], -1)
        x = self.layer1(x)
        o1 = self.layer2_1(x).view(batch_size, -1)
        o2 = self.layer2_2(x).view(batch_size, -1)
        o3 = self.layer2_3(x).view(batch_size, -1)
        out = torch.cat((o1,o2,o3), -1)
        print(out.shape)
        return out
        
        
class Anchor_CNN_LSTM(nn.Module):
    def __init__(self,
                 use_lstm=True,
                 use_cnn=True,
                 use_cnn2=False):
        super().__init__()
        
        out_dim = 0
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.use_cnn2 = use_cnn2
        
        if use_lstm == True:
            self.lstm_model = Anchor_LSTM_Model()
            out_dim += 800
            
        if use_cnn2:
            self.cnn_model = Anchor_CNN_Transformer()
            out_dim += 1792
        elif use_cnn == True:
            self.cnn_model = Anchor_CNN_Model()
            out_dim += 1536
        
        
        self.classifier = nn.Sequential(
                            nn.Linear(out_dim, 512),
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

    def forward(self, x_cnn, x_rnn):
        if self.use_lstm == True and self.use_cnn == True:
            x2 = self.cnn_model(x_cnn)
            x1 = self.lstm_model(x_rnn)
            x = torch.cat((x2, x1), -1)
        elif self.use_lstm == True:
            x = self.lstm_model(x_rnn)
        elif self.use_cnn == True or self.use_cnn2:
            x = self.cnn_model(x_cnn)

        out = self.classifier(x)

        return out
