import torch
import torch.nn as nn
import torch.nn.functional as F

fams = ['PKA', 'AKT', 'CDK', 'MAPK', 'SRC', 'CK2', 'PKC', 'PIKK']

class Model(nn.Module):
    def __init__(self,conv_drpt=0.0,mlp_drpt=0.0):
        super(Model, self).__init__()

        #### MOTIF NET ####
        self.conv1 = nn.Conv1d(22, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.penult = nn.Linear(64, 32)

        #### COORD NET ####
        self.mlp1 = nn.Linear(100, 112)
        self.bn1 = nn.BatchNorm1d(112)
        self.mlp2 = nn.Linear(112, 96)
        self.bn2 = nn.BatchNorm1d(96)
        self.mlp3 = nn.Linear(96, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.penult = nn.Linear(64, 32)

        ### CAT LAYERS ###
        self.penult = nn.Linear(64, 32)
        self.out = nn.Linear(32, len(fams))
        self.sigmoid = nn.Sigmoid()

        #### MISC LAYERS ####
        self.relu = nn.ReLU()
        self.conv_drpt = nn.Dropout(p = conv_drpt)
        self.mlp_drpt = nn.Dropout(p = mlp_drpt)
        self.ablate = nn.Dropout(p = 1.0)

    def forward(self, oneHot_motif, coords, version='seq-coord'):
        #### MOTIF NET ####
        conv1 = self.conv1(oneHot_motif.float())
        conv1 = self.relu(conv1)
        conv1 = self.pool(conv1)
        conv2 = self.conv2(conv1)
        conv2 = self.relu(conv2)
        conv2 = self.pool(conv2)
        conv3 = self.conv3(conv2)
        conv3 = self.relu(conv3)
        conv3 = self.pool(conv3)
        seq_out = conv3.view(conv3.size()[0], -1)
        seq_out = self.penult(seq_out) ## SEQ PENULT
        seq_out = self.relu(seq_out)
        seq_out = self.conv_drpt(seq_out)

        #### COORD NET ####
        mlp1 = self.mlp1(coords)
        mlp1 = self.relu(mlp1)
        mlp1 = self.bn1(mlp1)
        mlp1 = self.mlp_drpt(mlp1)
        mlp2 = self.mlp2(mlp1)
        mlp2 = self.relu(mlp2)
        mlp2 = self.bn2(mlp2)
        mlp2 = self.mlp_drpt(mlp2)
        mlp3 = self.mlp3(mlp2)
        mlp3 = self.relu(mlp3)
        mlp3 = self.bn3(mlp3)
        mlp3 = self.mlp_drpt(mlp3)
        coord_out = self.penult(mlp3)
        coord_out = self.relu(coord_out)

        if version=='seq-coord':
            seq_out = self.conv_drpt(seq_out) # seqCoord version
        else:
            seq_out = self.ablate(seq_out) # seq-only version
        
        coords_out = self.mlp_drpt(coord_out)
        
        cat = torch.cat((seq_out,coords_out), 1)
        cat = self.penult(cat)
        out = self.out(cat)
        out = self.sigmoid(out)
        
        return out