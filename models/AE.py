import torch
from torchvision import datasets
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F 

from torchvision import models
resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
                       "ResNet101": models.resnet101, "ResNet152": models.resnet152}
def get_para_num(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class ResNetEncoder(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetEncoder, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4

        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4,
                                            self.avgpool
                                            )
        self.in_features=2048*1*1
        # self.use_bottleneck = use_bottleneck
        # self.new_cls = new_cls
        # if new_cls:
        #     if self.use_bottleneck:
        #         self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        #         # self.fc = nn.Linear(bottleneck_dim, class_num)
        #         self.bottleneck.apply(init_weights)
        #         # self.fc.apply(init_weights)
        #         self.__in_features = bottleneck_dim
            # else:
                # self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                # self.fc.apply(init_weights)
                # self.__in_features = model_resnet.fc.in_features
        # else:
            # self.fc = model_resnet.fc
            # self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        # print("encoder",x.shape)
        x = x.view(x.size(0), -1)
        # if self.use_bottleneck and self.new_cls:
        #     x = self.bottleneck(x)
        # y = self.fc(x)
        return x

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_cls:
            if self.use_bottleneck:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                                  {"params": self.bottleneck.parameters(), "lr_mult": 10, 'decay_mult': 2}, \
                                  # {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}
                                  ]
            else:
                parameter_list = [{"params": self.feature_layers.parameters(), "lr_mult": 1, 'decay_mult': 2}, \
                                  # {"params": self.fc.parameters(), "lr_mult": 10, 'decay_mult': 2}
                                  ]
        else:
            parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class ResNetDecoder(nn.Module):
    def __init__(self, bottleneck_dim=256, latent_dim=[2048, 1, 1]):
        super(ResNetDecoder, self).__init__()
        self.lin2 = nn.Linear(bottleneck_dim, latent_dim[0] * latent_dim[1] * latent_dim[2])
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(latent_dim[0] * latent_dim[1] * latent_dim[2])
        self.drop=nn.Dropout(0.5)

        self.lin2.apply(init_weights)
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.t_conv1 = nn.ConvTranspose2d(2048, 256, kernel_size=3, stride=1)#size +7
        self.t_conv11 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2)  # size +7
        self.t_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)#size *2
        self.t_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)#size *2
        self.t_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)#size *2
        self.t_conv5 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)#size *2
        self.t_conv6 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)#size *2

        self.t_conv1.apply(init_weights)
        self.t_conv11.apply(init_weights)
        self.t_conv2.apply(init_weights)
        self.t_conv3.apply(init_weights)
        self.t_conv4.apply(init_weights)
        self.t_conv5.apply(init_weights)
        self.t_conv6.apply(init_weights)

        self.t_conv=nn.Sequential(
        self.t_conv1,nn.ReLU(),nn.BatchNorm2d(256),nn.Dropout(0.3),
        self.t_conv11,nn.ReLU(),nn.BatchNorm2d(256),nn.Dropout(0.3),
        self.t_conv2,nn.ReLU(),nn.BatchNorm2d(128),nn.Dropout(0.3),
        self.t_conv3,nn.ReLU(),nn.BatchNorm2d(128),nn.Dropout(0.3),
        self.t_conv4,nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(0.3),
        self.t_conv5,nn.ReLU(),nn.BatchNorm2d(64),nn.Dropout(0.3),
        self.t_conv6,
        nn.Sigmoid()
                )

    def forward(self, x):
        x = self.drop(self.bn1(self.relu(self.lin2(x))))
        x = x.view(-1, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2])
        x=self.t_conv(x)
        # print("decoder:",x.shape)
        return x



class LeNetEncoder(nn.Module):
    def __init__(self,use_bottleneck=True, bottleneck_dim=256):
        super(LeNetEncoder,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        # self.bn1=nn.BatchNorm2d(20),
        self.conv2 = nn.Conv2d(20, 50, 5)
        # self.bn2=nn.BatchNorm2d(50),
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = nn.Sequential(
            self.conv1,
            # self.bn1,
            nn.Dropout2d(0.1),
            nn.ReLU(),
            self.pool,
            self.conv2,
            # self.bn2,
            nn.Dropout2d(0.3),
            nn.ReLU(),
            self.pool
        )
        self.latent_dim=[50,4,4]
        self.bottleneck_dim=bottleneck_dim
        self.lin1 = nn.Linear(800, 256)
        # self.lin2 = nn.Linear(256, 800)
        self.lin1.apply(init_weights)
        self.bottle = nn.Sequential(
            self.lin1,
            nn.BatchNorm1d(self.bottleneck_dim, affine=True),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

    def forward(self,x):
        x=self.encoder(x)
        x=self.bottle(x.view(x.size(0), -1))
        return  x

class LeNetDecoder(nn.Module):
    def __init__(self, bottleneck_dim=256,latent_dim=[50,4,4]):
        super(LeNetDecoder,self).__init__()
        self.lin2 = nn.Linear(bottleneck_dim, latent_dim[0]*latent_dim[1]*latent_dim[2])
        self.lin2.apply(init_weights)
        self.latent_dim=latent_dim
        self.bottleneck_dim=bottleneck_dim
        self.t_conv1 = nn.ConvTranspose2d(50, 40, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(40, 20, kernel_size=5)
        self.t_conv3 = nn.ConvTranspose2d(20, 10, kernel_size=2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(10, 1, kernel_size=5)
        self.t_conv1.apply(init_weights)
        self.t_conv2.apply(init_weights)
        self.t_conv3.apply(init_weights)
        self.t_conv4.apply(init_weights)


    def forward(self,x):
        x = self.lin2(x)
        x = x.view(-1, self.latent_dim[0], self.latent_dim[1], self.latent_dim[2])
        x = torch.relu(self.t_conv1(x))
        x = torch.relu(self.t_conv2(x))
        x = torch.relu(self.t_conv3(x))
        x = torch.sigmoid(self.t_conv4(x))
        return  x

class ConvAutoencoder(nn.Module):
    def __init__(self,encoder_type="LeNet",bottleneck_dim=256,use_bootleneck=True):
        super(ConvAutoencoder, self).__init__()
        if "LeNet" in encoder_type:
            self.encoder=LeNetEncoder(use_bottleneck=use_bootleneck, bottleneck_dim=bottleneck_dim)
            self.decoder=LeNetDecoder(bottleneck_dim,latent_dim=self.encoder.latent_dim)
        elif "ResNet" in encoder_type:
            self.encoder=ResNetEncoder(resnet_name=encoder_type,use_bottleneck=use_bootleneck,
                                       bottleneck_dim=bottleneck_dim,new_cls=True,class_num=10)
            self.bottleneck=feat_bootleneck(self.encoder.in_features, bottleneck_dim=bottleneck_dim, type="bn")
            self.decoder=ResNetDecoder(bottleneck_dim=bottleneck_dim)
        # print(self.encoder)
        # print("encoder para:",get_para_num(self.encoder))
        # print("bottleneck para:", get_para_num(self.bottleneck))
        # print("decoder para:",get_para_num(self.decoder))
    def forward(self, x):

        x=self.encoder(x)
        x=self.bottleneck(x)
        z = x
        x=self.decoder(x)

        return x,None,None,z # none variational, both mu and var are None

    def loss_criterion(self, recon_x, x, mu=None, logvar=None):
        # print("re",recon_x.shape,"x",x.shape)
        # mse=F.mse_loss(recon_x,x,reduction="sum")
        bce = F.binary_cross_entropy(recon_x, torch.sigmoid(x), reduction='mean')
        # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce

    def get_feature_weight(self,recon_x,x):
        assert  len(recon_x) ==len(x)
        # assert  len(fea)==len(recon_x) and len(fea)==len(x)
        bce = F.binary_cross_entropy(recon_x, torch.sigmoid(x), reduction='none')
        bce=torch.mean(bce.view(bce.shape[0],-1),dim=1,keepdim=True)

        # normalize
        min_v = torch.min(bce)
        range_v = torch.max(bce) - min_v
        normalised_bce = (bce - min_v) / range_v

        #more loss, less weight
        weight=1-normalised_bce


        reweight=x.shape[0]*(weight/torch.sum(weight))
        # print(reweight.shape,reweight)

        # weighted_fea=fea*bce
        return reweight

    def get_finetune_modules(self):
        return [self.encoder,self.bottleneck]

    def gen_embedding(self, x):

        x = self.encoder(x)
        x=self.bottleneck(x)
        # x = x.view(x.size(0), -1)

        return x
#
#
#
# class ConvDenoisingAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvDenoisingAutoencoder, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
#
#         self.conv3 = nn.Conv2d(32, 64, 5, stride=5)
#         self.conv4 = nn.Conv2d(64, 128, 5, stride=5)
#
#         self.pool = nn.MaxPool2d(2, 2)
#
#         self.t_conv1 = nn.ConvTranspose2d(128, 64, 5, stride=5)
#         self.t_conv2 = nn.ConvTranspose2d(64, 32, 5, stride=5)
#
#         self.t_conv3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
#         self.t_conv4 = nn.ConvTranspose2d(16, 3, 2, stride=2)
#
#     def forward(self, x):
#         # add noise
#         x_noisy = x + x.data.new(x.size()).normal_(0, 0.1).type_as(x)
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#
#         x = torch.relu(self.conv3(x))
#         x = torch.relu(self.conv4(x))
#
#         x = torch.relu(self.t_conv1(x))
#         x = torch.relu(self.t_conv2(x))
#         x = torch.relu(self.t_conv3(x))
#         x = torch.sigmoid(self.t_conv4(x))
#
#         return x
#
#     def gen_embedding(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         x = self.pool(x)
#
#         x = torch.relu(self.conv3(x))
#         x = torch.relu(self.conv4(x))
#
#         x = x.view(x.size(0), -1)
#
#         return x

class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="bn"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(p=0.1)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            # x = self.dropout(x)
        return x

import torch.nn.utils.weight_norm as weightNorm

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="wn"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            # one layer
            self.fc1 = nn.Linear(bottleneck_dim, class_num)

            #two layer
            # self.fc1 = nn.Linear(bottleneck_dim, bottleneck_dim//2)
            # self.fc2 = nn.Linear(bottleneck_dim//2,class_num )
            # self.fc3 = nn.Linear(bottleneck_dim//2, class_num)
        elif type =='wn':
            #one layer
            # self.fc1=nn.Linear(bottleneck_dim, class_num)

            # self.normfc = weightNorm(self.fc1)

            #two layer
            # self.fc1 = nn.Linear(bottleneck_dim, class_num)
            # self.fc2 = nn.Linear(bottleneck_dim // 2, bottleneck_dim // 4)
            self.fc1 = weightNorm(nn.Linear(bottleneck_dim, class_num),name="weight")

            # self.normfc = (self.fc3)

        # self.bn1 =

        # self.relu = nn.ReLU(inplace=True)

        # self.dropout = nn.Dropout(p=0.5)
        self.fc1.apply(init_weights)
        # self.fc2.apply(init_weights)
        # self.fc3.apply(init_weights)

        self.linears=nn.Sequential(

                                  self.fc1,
                                  # self.fc1,self.relu,nn.BatchNorm1d(bottleneck_dim//2,affine=True),nn.Dropout(0.3),
                                  # self.fc2, self.relu, nn.BatchNorm1d(bottleneck_dim // 4, affine=True), nn.Dropout(0.3),
                                  # self.fc3,
                                  # self.fc2,
                                  # self.fc3,
                                  )
    def forward(self, x):
        # print(x.shape)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x=F.normalize(x)
        x=self.linears(x)
        return x


class ConvVariationalAutoencoder(nn.Module):
    def __init__(self,encoder_type="LeNet",bottleneck_dim=256,use_bootleneck=True):
        super(ConvVariationalAutoencoder, self).__init__()


        if "LeNet" in encoder_type:
            self.encoder = LeNetEncoder(use_bottleneck=use_bootleneck, bottleneck_dim=bottleneck_dim)
            self.decoder = LeNetDecoder(bottleneck_dim, latent_dim=self.encoder.latent_dim)
            self.latent_dim = 800
        elif "ResNet" in encoder_type:
            self.encoder = ResNetEncoder(resnet_name=encoder_type, use_bottleneck=use_bootleneck,
                                         bottleneck_dim=bottleneck_dim, new_cls=True, class_num=10)
            # self.bottleneck = feat_bootleneck(self.encoder.in_features, bottleneck_dim=bottleneck_dim, type="bn")

            self.latent_dim = 2048
            self.decoder = ResNetDecoder(bottleneck_dim=self.latent_dim)

        self.trans_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.trans_var = nn.Linear(self.latent_dim, self.latent_dim)
        print(self.encoder)
        print("encoder para:", get_para_num(self.encoder))
        # print("bottleneck para:", get_para_num(self.bottleneck))
        print("decoder para:", get_para_num(self.decoder))
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):


        x=self.encoder(x)
        mu = self.trans_mu(x.view(-1, self.latent_dim))
        logvar = self.trans_var(x.view(-1, self.latent_dim))
        z = self.reparameterize(mu, logvar)
        x=self.decoder(z)

        return x, mu, logvar,z

    
    def loss_criterion(self, recon_x, x, mu, logvar):
        # print("re",recon_x.shape,"x",x.shape)
        # mse=F.mse_loss(recon_x,x,reduction="sum")
        bce = F.binary_cross_entropy(recon_x, torch.sigmoid(x), reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return bce + kld

    def get_feature_weight(self,recon_x,x):
        assert  len(recon_x) ==len(x)
        # assert  len(fea)==len(recon_x) and len(fea)==len(x)
        bce = F.binary_cross_entropy(recon_x, torch.sigmoid(x), reduction='none')
        bce=torch.mean(bce.view(bce.shape[0],-1),dim=1,keepdim=True)

        # normalize
        min_v = torch.min(bce)
        range_v = torch.max(bce) - min_v
        normalised_bce = (bce - min_v) / range_v

        #more loss, less weight
        weight=1-normalised_bce


        reweight=x.shape[0]*(weight/torch.sum(weight))
        # print(reweight.shape,reweight)

        # weighted_fea=fea*bce
        return reweight

    def get_finetune_modules(self):

        return [self.encoder,nn.Sequential(self.trans_mu,self.trans_var)]

    def gen_embedding(self, x):
        x=self.encoder(x)
        mu = self.trans_mu(x.view(-1, self.latent_dim))
        logvar = self.trans_var(x.view(-1, self.latent_dim))
        z = self.reparameterize(mu, logvar)
        return z


# # train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
# # test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
#
# # batch_size = 512
# # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0)
# # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
#
# # model = ConvAutoencoder().cuda()
# # model = ConvDenoisingAutoencoder().cuda()
# model = ConvVariationalAutoencoder().cuda()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
#
#
# def train(epochs):
#     for epoch in range(1, epochs + 1):
#         train_loss = 0
#         for _ in range(10):
#             images = torch.randn(512, 200, 100, 3)
#             images = images.transpose(3, 2).transpose(2, 1)
#             images = images.cuda()
#             optimizer.zero_grad()
#
#             # For Vanilla/Denoising autoencoder
#             # outputs = model(images)
#             # loss = criterion(outputs, images)
#
#             # For variational autoencoder
#             outputs, mu, logvar = model(images)
#             loss = model.loss_criterion(outputs, images, mu, logvar)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item() * images.size(0)
#         print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
#         if epoch == epochs:
#             images = torch.randn(512, 200, 100, 3)
#             images = images.transpose(3, 2).transpose(2, 1)
#             images = images.cuda()
#             embedding = model.gen_embedding(images)
#             print(embedding.size())
#
#
# if __name__ == '__main__':
#     train(10)

