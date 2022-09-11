import torch.nn as nn
import torch
class UNet(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
      """

    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        u_out = self.final(u45)
        return  u_out

class LinearNet(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.real = nn.Linear(N, N, bias=False)
        self.imag = nn.Linear(N, N, bias=False)

    def forward(self, x):
        def matmul_complex(t1, t2):
            return torch.view_as_complex(
                torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real), dim=1))

        fft_in_rows = torch.zeros([len(x), len(x)]).to(torch.cfloat).to("cuda:0")
        fft_in_cols = torch.zeros([len(x), len(x)]).to(torch.cfloat).to("cuda:0")
        weights = self.real.state_dict()["weight"] + 1j * self.imag.state_dict()["weight"]
        for i in range(len(x)):
            row_fft_real = self.real(x[i, :])
            row_fft_imag = self.imag(x[i, :])
            row_fft = row_fft_real + 1j * row_fft_imag
            fft_in_rows[i, :] = fft_in_rows[i, :] + row_fft
        for i in range(len(x)):
            col_fft = matmul_complex(fft_in_rows[:, i], weights)
            fft_in_cols[:, i] = fft_in_cols[:, i] + col_fft
        res = torch.cat([fft_in_cols.real, fft_in_cols.imag],dim=1)
        return res
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Combine(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, N,in_channels=3, out_channels=3):
        super(Combine, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )
        self.real = nn.Linear(N, N, bias=False)
        self.imag = nn.Linear(N, N, bias=False)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        u_out = self.final(u45)
        def matmul_complex(t1, t2):
            return torch.view_as_complex(
                torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real), dim=1))

        res_list = []
        for i in range(len(u_out)):
            u_out_batch = u_out[i,:,:,:]
            rgb_list = []
            for c in range(3):
                u_out_tmp = u_out_batch[c,:,:]
                fft_in_rows = torch.zeros([len(u_out_tmp), len(u_out_tmp)]).to(torch.cfloat).to("cuda:0")
                fft_in_cols = torch.zeros([len(u_out_tmp), len(u_out_tmp)]).to(torch.cfloat).to("cuda:0")
                weights = self.real.state_dict()["weight"] + 1j * self.imag.state_dict()["weight"]
                for j in range(len(u_out_tmp)):
                    row_fft_real = self.real(u_out_tmp[j, :])
                    row_fft_imag = self.imag(u_out_tmp[j, :])
                    row_fft = row_fft_real + 1j * row_fft_imag
                    fft_in_rows[j, :] = fft_in_rows[j, :] + row_fft
                for k in range(len(u_out_tmp)):
                    col_fft = matmul_complex(fft_in_rows[:, k], weights)
                    fft_in_cols[:, k] = fft_in_cols[:, k] + col_fft
                res = torch.cat([fft_in_cols.real, fft_in_cols.imag], dim=1)
                rgb_list.append(res)
            rgb_list = torch.stack(rgb_list)
            res_list.append(rgb_list)
        res_list = torch.stack(res_list)
        return u_out,res_list