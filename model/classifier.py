import torch 
from torch import nn

class Classifier(nn.Module):
    def __init__(self, n_way):
      super().__init__() 
      self.layer = nn.Sequential(nn.Flatten(),
                                    nn.Linear(128, n_way),
                                    nn.Softmax(dim=1))
      self.initilize()
 
    def forward(self, inputs, norm=False): 
      if norm:
         inputs = self.l2_norm(inputs)
      x = self.layer(inputs)
      return x
    
    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


