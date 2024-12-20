%%writefile models.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv_output_size, mi_bce_loss

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.l2 = nn.Linear(32, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        clazz = F.relu(self.bn1(self.l1(x)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# class Encoder(nn.Module):
#     """
#     DCGAN Discriminator architecture from (add github repo url)

#     Except use ReLU instead of LeakyReLu, and batch norm after all hidden layers
#     Removed single conv layer with stride since input is 32x32 instead of 64x64
#     Single hidden layer w/ 1024 units after last conv layer
#     """
#     def __init__(self,
#             input_shape=(3, 32, 32),
#             encoding_size=64,
#             feature_layer=3):

#         super().__init__()

#         self.encoding_size = encoding_size

#         in_channels, H, W = input_shape
#         out_channels = 64

#         lfC, lfH, lfW = input_shape # initial shape of local feature

#         layers = []
#         for i in range(2):
#             layers += [ConvLayer(in_channels, out_channels, 4, 2, 1, bias=False)]
#             H, W = conv_output_size((H,W), 4, 2, 1)
#             if i+1 == feature_layer:    # Save the shape of the local feature layer
#                 lfC, lfH, lfW = out_channels, H, W
#             in_channels, out_channels = out_channels, out_channels * 2

#         self.local_feature_shape = (lfC, lfH, lfW)
#         #print(f'C {out_channels//2}, H {H}, W {W}')
#         #print(f'lfC {lfC}, lfH {lfH}, lfW {lfW}')
#         # Local feature encoder
#         self.C = nn.Sequential(*layers[:feature_layer])
#         print(self.C)
#         # Global feature encoder
#         self.f0 = nn.Sequential(*layers[feature_layer:])    # Remaining conv layers
#         print(self.f0)
#         self.f1 = nn.Sequential(
#                 nn.Linear(out_channels//4 * H * W, 256, bias=False),
#                 nn.BatchNorm1d(256),
#                 nn.ReLU()
#             )

#         self.linear = nn.Linear(256, self.encoding_size, bias=True)

#     def conv(x):
#         return self.C(self.f0(x))

#     def fc(x):
#         return self.f1(self.conv(x))

#     def forward(self, x):
#         M = self.C(x)
#         print(M.shape)                             # Local feature
#         conv = self.f0(M)                           # Last conv Layer
#         fc = self.f1(conv.view(conv.size(0), -1))   # fully-connected
#         y = self.linear(fc)                         # Global encoding
#         return y, M


class Encoder(nn.Module):
    def __init__(self, input_shape=(3, 32, 32),
                        encoding_size=64,
                        feature_layer=3):
        super(Encoder, self).__init__()
        self.first_encoder=nn.Sequential(

            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

        )


        self.second_encoder=nn.Sequential(

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.local_feature_shape = (128, 8, 8)
        self.encoding_size = encoding_size
        self.fc1 = nn.Linear(256, self.encoding_size)
        # self.fc2 =nn.Linear(256,256)
        self.init_weights()

    def init_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight)
          nn.init.constant_(m.bias,0)

    def forward(self, x):
        # print("----->",x.shape)
        M = self.first_encoder(x)
        #print(first_feature.shape)
        second_feature = self.second_encoder(M)
        # print(second_feature.shape)
        output = F.max_pool2d(second_feature, 4).view(-1, 256)
        # print("--->",output.shape)

        y = self.fc1(output)
        # print(y.shape)
        # z_var=self.fc2(output)
        return y, M



class GlobalDIM(nn.Module):
    def __init__(self,
            encoding_size=64,
            local_feature_shape=(128, 8, 8)):
        super().__init__()
        in_features = encoding_size + np.product(local_feature_shape)

        self.main = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )

    def forward(self, y, M):
        # Flatten local features and concat with global encoding
        x = torch.cat([M.view(M.size(0), -1), y], dim=-1)
        return self.main(x)


class ConcatAndConvDIM(nn.Module):
    """ Concat and Convolve Architecture """
    def __init__(self,
            encoding_size=64,
            local_feature_shape=(128, 8, 8)):
        super().__init__()

        c, lH, lW = local_feature_shape

        self.sequential = nn.Sequential(
                nn.Conv2d(c + encoding_size, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(512, 1, kernel_size=1)
            )

    def forward(self, y, M):
        b, lC, lH, lW = M.shape
        # Concat
        y = y.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, lH, lW)
        x = torch.cat([M, y], dim=1)    # Over channel dimension
        # Convolve
        return self.sequential(x)

class BlockLayerNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs):
        x = inputs.permute(0,2,3,1)
        x = self.layer_norm(x)
        return x.permute(0,3,1,2)

class EncodeAndDotDIM(nn.Module):
    """ Encode and Dot Architecture """
    def __init__(self,
            encoding_size=64,
            local_feature_shape=(128, 8, 8)):

        super().__init__()

        lC, lH, lW = local_feature_shape

        # Global encoder
        self.G1 = nn.Sequential(
                nn.Linear(encoding_size, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(),
                nn.Linear(2048, 2048)
            )
        self.G2 = nn.Sequential(
                nn.Linear(encoding_size, 2048),
                nn.ReLU()
            )

        # Local encoder
        self.block_layer_norm = BlockLayerNorm(2048)
        self.L1 = nn.Sequential(
                nn.Conv2d(lC, 2048, 1, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.Conv2d(2048, 2048, 1, bias=True)
            )

        self.L2 = nn.Sequential(
                nn.Conv2d(lC, 2048, 1),
                nn.ReLU(),
            )

    def forward(self, y, M):
        g = self.G1(y) + self.G2(y)
        l = self.block_layer_norm(self.L1(M) + self.L2(M))

        # broadcast over channel dimension
        g = g.view(g.size(0), g.size(1), 1, 1)
        return (g * l).sum(1)



class PriorMatch(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.sequential = nn.Sequential(
                nn.Linear(in_features, 1000, bias=False),
                nn.BatchNorm1d(1000),
                nn.ReLU(),
                nn.Linear(1000, 200, bias=False),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(200, 1)
            )

    def forward(self, x):
        return self.sequential(x)


class MIEstimator(nn.Module):
    def __init__(self,
            alpha, beta, gamma,
            encoding_size=64,
            local_feature_shape=(128, 8, 8),
            encode_and_dot=True,
            num_negative=2):

        super().__init__()

        args = (encoding_size, local_feature_shape)
        # Number of negative samples
        if num_negative < 1:
            raise ValueError(f"Arg num_negative with value {num_negative} should be >= 1")
        self.num_negative = num_negative

        # Don't waste resources if hyperparameters are set to zero
        self.global_disc = GlobalDIM(*args) if alpha > 0 else None

        if encode_and_dot:
            self.local_disc = EncodeAndDotDIM(*args) if beta > 0 else None
        else:
            self.local_disc = ConcatAndConvDIM(*args) if beta > 0 else None

        self.prior_disc = PriorMatch(encoding_size) if gamma > 0 else None

        # Distributions won't move to GPU
        #self.prior = Uniform(torch.cuda.FloatTensor(0), torch.cuda.FloatTensor(1))

        # DIM Hyperparameters
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, y, M):
        """
        Args:
            y (torch.Tensor): Global encoding output of Encoder
            M (torch.Tensor): Local encoding output of Encoder
        """

        # Default values if loss isn't used
        global_loss = local_loss = prior_loss = 0.

        # Number of negative samples doesn't have a large effect
        # on JSD MI (implemented as BCE), so 1 negative sample is used
        # Rotate along batch dimension to create M_prime
        #M_prime = [torch.cat([M[i:], M[:i]], 0).detach() for i in range(len(M)-1)]
        # Negative index selection
        n = M.size(0)
        idx = list(range(n))
        neg_idx = [(idx[i+1:] + idx[:i])[:self.num_negative] for i in range(n)]
        M_prime = M[neg_idx, ...]   # Shape is [batch_size, batch_size-1, C, H, W]

        # Global MI loss
        if not self.global_disc is None:
            positive = self.global_disc(y, M)
            negative = torch.mean(torch.stack(
                [self.global_disc(y, M_prime[:,i]) for i in range(self.num_negative)]))
            #negative = self.global_disc(y, M_prime)
            global_loss = self.alpha * mi_bce_loss(positive, negative)

        # Local MI loss
        if not self.local_disc is None:
            lH, lW = M.shape[2:]
            positive = self.local_disc(y, M)
            negative = torch.mean(torch.stack(
                [self.local_disc(y, M_prime[:,i]) for i in range(self.num_negative)]))
            #negative = self.local_disc(y, M_prime)
            local_loss = self.beta * mi_bce_loss(positive, negative)/(lH*lW)

        # Prior (discriminator) loss
        if not self.prior_disc is None:
            prior_sample = torch.rand_like(y)
            positive = self.prior_disc(prior_sample)
            negative = self.prior_disc(torch.sigmoid(y))
            prior_loss = self.gamma * mi_bce_loss(positive, negative)

        return global_loss, local_loss, prior_loss

class _Encoder(nn.Module):
  def __init__(self):
      super(_Encoder, self).__init__()
      self.encoder = Encoder()
      self.FC_mean = nn.Linear(64, 64)
      self.FC_var = nn.Linear(64, 64)

  def forward(self, input):
      result = self.encoder(input)[0]
      # print("------------->", result.shape)
      result = torch.flatten(result, start_dim=1)
      # print("-------->", result.shape)
      mu = self.FC_mean(result)
      log_var = self.FC_var(result)

      return mu, log_var, result


class _Decoder(nn.Module):
    def __init__(self):
      super(_Decoder, self).__init__()
      # Number of channels in the training images. For color images this is 3
      nc = 3
      nz = 64
      ngf = 32

      self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, 3, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            # nn.ReLU(True),
            # # state size. ``(ngf) x 32 x 32``
            # nn.ConvTranspose2d( ngf, nc, 4, 1, 1, bias=False),
            # nn.Sigmoid()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, x):
      x = x.reshape(x.shape[0], x.shape[1], 1, 1)
      # print("-->", x.shape)
      x = self.decoder(x)
      # print("*-->", x.shape)
      x = torch.nn.functional.sigmoid(x)
      # x = nn.Sigmoid(x.view(x.shape[0], -1))
      # print("**-->", x.shape)

      # print(x)
      return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.Encoder = _Encoder()
        self.Decoder = _Decoder()

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)        # sampling epsilon
        z = mean + var*epsilon                          # reparameterization trick
        return z


    def forward(self, x):
        mean, log_var, encoded = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var




# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     torch.manual_seed(0)

#     # Encoder
#     x = torch.randn([4,3,32,32])
#     encoder = Encoder(feature_layer=2)
#     y, M = encoder(x)

#     # Global
#     y = torch.randn(4, 64)
#     M = torch.randn(4, 128, 8, 8)
#     global_disc = GlobalDIM(64, (128, 8, 8))
#     logits = global_disc(y, M)

#     # Local (concat and convolve)
#     y = torch.randn(4, 64)
#     M = torch.randn(4, 128, 8, 8)
#     local_disc = ConcatAndConvDIM(64, (128, 8, 8))
#     logits = local_disc(y, M)

#     # Local (encode and dot)
#     y = torch.randn(4, 64)
#     M = torch.randn(4, 128, 8, 8)
#     local_disc = EncodeAndDotDIM(64, (128, 8, 8))
#     logits = local_disc(y, M)

#     # Prior
#     z = torch.rand(64).to(device)
#     prior_disc = PriorMatch(in_features=64)
#     prior_disc.to(device)
#     prior_disc(z)

#     # MI Estimator
#     y = torch.rand(4,64)
#     M = torch.randn(4,128,8,8)
#     M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)

#     mi_estimator = MIEstimator(.5, .5, 1,
#             local_feature_shape=M.shape[1:])
#     global_loss, local_loss, prior_loss = mi_estimator(y, M, M_prime)
