import torch.nn as nn

class VODecoder(nn.Module):
    def __init__(self, feature_size):
        super(VODecoder, self).__init__()

        self.fc1 = nn.Linear(in_features=feature_size, out_features=128)
        self.tanh_1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.tanh_2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=128, out_features=6)
        self.rot = nn.Linear(in_features=128, out_features=9)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.normal_(self.rot.weight, 0, 0.01)
        nn.init.zeros_(self.rot.bias)

    def forward(self, input):

        x = self.fc1(input)
        x = self.tanh_1(x)
        x = self.fc2(x)
        x = self.tanh_2(x)
        p = self.fc3(x)
        r = self.rot(x)
        
        return p, r
