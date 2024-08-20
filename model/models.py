import torch 
from torch import nn

class chainModels(nn.Module):
 def __init__(self, extactor, classifier):
    super().__init__() 
    self.extractor = extactor
    self.classifier = classifier
 
 def forward(self,inputs): 
    x = self.extractor(inputs)
    x = self.classifier(x)
    return x