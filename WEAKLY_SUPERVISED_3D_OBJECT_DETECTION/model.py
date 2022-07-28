import imp
from turtle import forward
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F


class WSDNN_Resnet(nn.Module):
    def __init__(self, 
                roi_size=(16, 5),
                num_class=1):
        super(WSDNN_Resnet, self).__init__()
        self.roi_size = roi_size
        self.num_class = num_class
        resnet = torchvision.models.resnet152(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        self.encoder = torch.nn.Sequential(*(list(resnet.children())[:-3]))

        self.roi_pool = torchvision.ops.roi_align
        self.classifier = nn.Sequential(
                        nn.Linear(1024*self.roi_size[0]*self.roi_size[1], 4096), 
                        nn.ReLU(inplace=True), 
                        nn.Linear(4096, 4096), 
                        nn.Dropout(p=0.5),
                        nn.ReLU(inplace=True))

        self.score_fc = nn.Sequential(
                            nn.Linear(4096, self.num_class))
                            #nn.Softmax(dim=1))

        self.bbox_fc = nn.Sequential(
                            nn.Linear(4096, self.num_class))
                            #nn.Softmax(dim=0))
        self.dropout = nn.Dropout() 

    def forward(self, x, rois):
        #with torch.no_grad():
        out = self.encoder(x)
        h, w = out.shape[2], out.shape[3]
        spp_output = self.roi_pool(out, [rois], self.roi_size, h/x.shape[2])
        spp_output = spp_output.reshape(spp_output.shape[0], -1)
        classifier_ouput = self.classifier(spp_output)
        
        class_scores = self.dropout(self.score_fc(classifier_ouput))
        class_scores = torch.softmax(class_scores, 0)

        bbox_scores = self.dropout(self.bbox_fc(classifier_ouput))
        bbox_scores = torch.sigmoid(bbox_scores)

        cls_prob = class_scores * bbox_scores 
        return cls_prob.unsqueeze(1)


class WSDNN_Alexnet(nn.Module):
    def __init__(self, 
                roi_size=(6, 6),
                num_class=1):
        super(WSDNN_Alexnet, self).__init__()
        self.roi_size = roi_size
        self.num_class = num_class
        #resnet = torchvision.models.alexnet(True)

        # for param in resnet.parameters():
        #     param.requires_grad = False

        self.features = torchvision.models.alexnet(pretrained=True).features
        for name in self.features.modules():
            name[12] = nn.Identity()
            break
        # for i, (name,param) in enumerate(self.features.named_parameters()):
        #     param.requires_grad = False
        #     print(name, param.requires_grad)
        #self.encoder = torch.nn.Sequential(*(list(resnet.children())[:-2]))

        self.roi_pool = torchvision.ops.roi_pool
        self.classifier = nn.Sequential(
                        nn.Linear(256*self.roi_size[0]*self.roi_size[1], 4096), 
                        nn.ReLU(inplace=True), 
                        nn.Linear(4096, 4096), 
                        nn.ReLU(inplace=True))

        self.score_fc   = nn.Sequential(
                            nn.Linear(4096, self.num_class))
                            #nn.Softmax(dim=1))

        self.bbox_fc    = nn.Sequential(
                            nn.Linear(4096, self.num_class))
                            #nn.Softmax(dim=0))
        
    def forward(self, x, rois, isval=False):
        #with torch.no_grad():
        out = self.features(x)
        h, w = out.shape[2], out.shape[3]
        spp_output = self.roi_pool(out, [rois], self.roi_size, h/x.shape[2])
        spp_output = spp_output.reshape(spp_output.shape[0], -1)
        classifier_ouput = self.classifier(spp_output)
        
        class_scores = self.score_fc(classifier_ouput)
        class_scores = torch.softmax(class_scores, 0)

        bbox_scores = self.bbox_fc(classifier_ouput)
        bbox_scores = torch.sigmoid(bbox_scores)

        cls_prob = class_scores * bbox_scores 
        if isval:
            return cls_prob.unsqueeze(1), class_scores.unsqueeze(1)
        else:
            return cls_prob.unsqueeze(1)

if __name__ == '__main__':
    model = WSDNN_Resnet()
    x = torch.randn((1, 3, 1224, 370))
    with torch.no_grad():
        out = model(x)
        print(out.shape)