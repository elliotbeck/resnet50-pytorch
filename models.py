import torchvision.models as models
from torch import nn

class resnet(nn.Module):
   def __init__(self, hidden_dim, num_classes):
      super(resnet, self).__init__()
      self.num_classes = num_classes
      self.resnet50 = models.resnet50(pretrained=True)
      # for param in self.resnet50.parameters():
      #    param.requires_grad = False
      self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
      # self.linear1 = nn.Linear(hidden_dim, hidden_dim)
      # self.linear2 = nn.Linear(hidden_dim, num_classes)
      # self.dropout = nn.Dropout(0.5)
      # self.relu = nn.ReLU(True)

   def logits(self, input):
      x = self.resnet50(input)
      # x = self.relu(x)
      # x = self.dropout(x)
      # x = self.linear1(x)
      # x = self.relu(x)
      # x = self.dropout(x)
      # x = self.linear2(x)
      return x

   def forward(self, input):
      input = input.permute(0,3,1,2).cuda()
      x = self.logits(input)
      return x