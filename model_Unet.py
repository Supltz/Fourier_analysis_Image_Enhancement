import torch.nn as nn
import torch
class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(UNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.r_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.rr_dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.r_dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)
             
        self.r_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.r_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.r_up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.r_up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        

        self.i_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.i_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.i_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.i_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.i_dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.i_dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.i_dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.i_dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.ii_dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.i_dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)
             
        self.i_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.i_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.i_up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.i_up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                # nn.Dropout(0.1),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            # nn.Dropout(0.1),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
                #nn.Softsign())
        return layer

    def forward(self, x):
        x_in = torch.cat((x, x), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        r_d0 = torch.cat((self.r_up1(e4), e3), 1)

        r_d0 = self.r_dc1(r_d0)
        r_d0 = self.r_dc2(r_d0)

        r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)

        r_d1 = self.r_dc3(r_d1)
        r_d1 = self.r_dc4(r_d1)

        r_d2 = torch.cat((self.r_up3(r_d1), e1), 1)

        r_d2 = self.r_dc5(r_d2)
        r_d2 = self.r_dc6(r_d2)

        r_d3 = torch.cat((self.r_up4(r_d2), e0), 1)
        r_d3 = self.r_dc7(r_d3)
        r_d3 = self.r_dc8(r_d3)

        f_r = self.rr_dc9(r_d3)
        # print('r_d2.shape   ', r_d2.shape)
        # f_r = self.rr_dc9(r_d2)
        

        i_d0 = torch.cat((self.i_up1(e4), e3), 1)

        i_d0 = self.i_dc1(i_d0)
        i_d0 = self.i_dc2(i_d0)

        i_d1 = torch.cat((self.i_up2(i_d0), e2), 1)

        i_d1 = self.i_dc3(i_d1)
        i_d1 = self.i_dc4(i_d1)

        i_d2 = torch.cat((self.i_up3(i_d1), e1), 1)

        i_d2 = self.i_dc5(i_d2)
        i_d2 = self.i_dc6(i_d2)

        i_d3 = torch.cat((self.i_up4(i_d2), e0), 1)
        i_d3 = self.i_dc7(i_d3)
        i_d3 = self.i_dc8(i_d3)

        f_i = self.ii_dc9(i_d3)
        # f_i = self.ii_dc9(i_d2)
        
        return f_r[:,0:1,:,:], f_i[:,0:1,:,:]#)#, torch.complex(f_r[:,1:2,:,:], f_i[:,1:2,:,:])