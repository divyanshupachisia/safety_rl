# Please contact the author(s) of this library if you have any questions.
# Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

import torch
import torch.nn as nn

class Sin(nn.Module):
    """
    Sin: Wraps element-wise `sin` activation as a nn.Module. 

    Shape:
        - Input: `(N, *)` where `*` means, any number of additional dimensions
        - Output: `(N, *)`, same shape as the input

    Examples:
        >>> m = Sin()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def __init__(self): 
        super().__init__() # init the base class

    def forward(self, input):
        return torch.sin(input) # simply apply already implemented sin


class model(nn.Module):
    """
    model: Constructs a fully-connected neural network with flexible depth, width
            and activation function choices.

    Args:
        dimList (int List): the dimension of each layer.
        actType (string): the type of activation function. Defaults to 'Tanh'.
                            currently supports 'Sin', 'Tanh' and 'ReLU'.
    """    
    def __init__(self, dimList, actType='Tanh'):
        super(model, self).__init__()

        # Construct module list: if use `Python List`, the modules are not added to
        # computation graph. Instead, we should use `nn.ModuleList()`.
        self.moduleList = nn.ModuleList()
        numLayer = len(dimList)-1
        for idx in range(numLayer):
            i_dim = dimList[idx]
            o_dim = dimList[idx+1]

            self.moduleList.append(nn.Linear(in_features=i_dim, out_features=o_dim))
            if idx == numLayer-1: # final linear layer, no act.
                pass
            else:
                if actType == 'Sin':
                    self.moduleList.append(Sin())
                elif actType == 'Tanh':
                    self.moduleList.append(nn.Tanh())
                elif actType == 'ReLU':
                    self.moduleList.append(nn.ReLU())
        print(self.moduleList)

        # Initalizes the weight
        self._initialize_weights()
        print("Using {:d}-layer NN architecture with {:s} act.".format(
            numLayer, actType))


    def forward(self, x):
        for m in self.moduleList:
            x = m(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


# ! Deprecated method, do not use
class modelTanhTwo(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelTanhTwo, self).__init__()
        # nn.ReLU() nn.Tanh()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=100),
            nn.Tanh())
        self.final = nn.Linear(100, action_num)
        self._initialize_weights()
        print("Using two-layer NN architecture with Tanh act.")

    def forward(self, x):
        out1 = self.fc1(x)
        a = self.final(out1)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


# ! Deprecated method, do not use
class modelTanhThree(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelTanhThree, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_num, out_features=100),
            nn.Tanh())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=100),
            nn.Tanh())
        self.final = nn.Linear(100, action_num)
        self._initialize_weights()
        print("Using three-layer NN architecture with Tanh act.")

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(out1)
        a = self.final(out2)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


# ! Deprecated method, do not use
class modelSinTwo(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelSinTwo, self).__init__()
        self.fc1 = nn.Linear(in_features=state_num, out_features=100)
        self.final = nn.Linear(100, action_num)
        self._initialize_weights()
        print("Using two-layer NN architecture with Sin act.")

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = torch.sin(out1)
        a = self.final(out1)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


# ! Deprecated method, do not use
class modelSinThree(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelSinThree, self).__init__()
        self.fc1 = nn.Linear(in_features=state_num, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.final = nn.Linear(100, action_num)
        self._initialize_weights()
        print("Using three-layer NN architecture with Sin act.")

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = torch.sin(out1)
        out2 = self.fc2(out1)
        out2 = torch.sin(out2)
        a = self.final(out2)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()


# ! Deprecated method, do not use
class modelSinFour(nn.Module):
    def __init__(self, state_num, action_num):
        super(modelSinFour, self).__init__()
        self.fc1 = nn.Linear(in_features=state_num, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=512)
        self.final = nn.Linear(512, action_num)
        self._initialize_weights()
        print("Using four-layer NN architecture with Sin act.")

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = torch.sin(out1)
        out2 = self.fc2(out1)
        out2 = torch.sin(out2)
        out3 = self.fc2(out2)
        out3 = torch.sin(out3)
        a = self.final(out3)
        return a

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 1)
                m.bias.data.zero_()