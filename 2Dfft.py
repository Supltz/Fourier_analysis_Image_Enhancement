import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import  math
from tqdm import tqdm
from model import *

def create_fourier_weights(signal_length):
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals/ signal_length
    return  np.cos(theta_vals)-1j*np.sin(theta_vals)




# grating_shifted = np.fft.ifftshift(grating)
# ft = np.fft.fft2(grating_shifted)
# ft = np.fft.fftshift(ft)
# plt.subplot(121)
# plt.imshow(abs(ft))
# plt.show()
#
# fft_in_rows = np.zeros([N,N],dtype=complex)
# for i in range(len(grating_shifted)):
#     row_fft = grating_shifted[i,:]@fft_weight
#     fft_in_rows[i,:] = fft_in_rows[i,:] + row_fft
# fft_in_cols = np.zeros([N,N],dtype=complex)
# for i in range(len(grating_shifted)):
#     col_fft = np.matmul(fft_in_rows[:,i],fft_weight)
#     fft_in_cols[:,i] = fft_in_cols[:,i] + col_fft
# fft = np.fft.fftshift(fft_in_cols)
# plt.subplot(122)
# plt.imshow(abs(fft))
# plt.show()
#
N = 64
fft_weight = create_fourier_weights(N)
data = "signals"
batchsize = 64
loss_func = nn.MSELoss()
device = 'cuda:0'
net = Net(N)
net.to(device)
optimizer = optim.Adam(net.parameters(),lr=0.01)

if data == "noise":
    for epoch in range(2500):
        noise = torch.rand(N,N).to(device)
        target = np.fft.ifftshift(noise.cpu())
        target = np.fft.fft2(target)
        target = np.fft.fftshift(target)
        target_comb = np.hstack([target.real,target.imag])
        target_combTensor = torch.Tensor(target_comb).to(device)
        optimizer.zero_grad()
        outputs = net(noise)
        loss = loss_func(outputs,target_combTensor)
        loss.backward()
        optimizer.step()
        print("epoch_{}: {}".format(epoch,loss.item()))
    #test
    x = np.arange(-N // 2, N // 2, 1)
    X, Y = np.meshgrid(x, x)
    wavelength = 20
    angle = 60
    grating = np.sin(
        2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength
    )
    target_test = np.fft.ifftshift(grating)
    target_test = np.fft.fft2(target_test)
    target_test = np.fft.fftshift(target_test)
    grating = torch.Tensor(grating).to(device)
    test_output = net(grating)
    test_output_real = test_output[:, 0:N]
    test_output_imag = test_output[:, N:2 * N]
    test_fft = test_output_real + 1j * test_output_imag
    ift = np.fft.ifftshift(test_fft.cpu().detach().numpy())
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    ift = ift.real
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(grating.cpu().numpy(), cmap="gray")
    plt.subplot(222)
    plt.title("Predict FFT")
    plt.imshow(np.log(abs(test_fft.cpu().detach().numpy())), cmap="gray")
    plt.subplot(223)
    plt.title("True FFT")
    plt.imshow(np.log(abs(target_test)), cmap="gray")
    plt.subplot(224)
    plt.title("Restored image")
    plt.imshow(ift, cmap="gray")
    plt.tight_layout()
    plt.show()
    ift = torch.Tensor(ift).to(device)
    loss_test = loss_func(grating, ift)
    print(loss_test.item())


elif data=="signals":
    epoches = []
    loss_value = []
    for epoch in range(8000):
        x = np.arange(-N // 2, N // 2, 1)
        X, Y = np.meshgrid(x, x)
        wavelength = np.round(np.random.rand(1)*10 + 1).astype(np.int32)
        angle = np.round(np.random.rand(1)*45).astype(np.int32)
        grating = np.sin(
            2 * np.pi * (X * np.cos(angle) + Y * np.sin(angle)) / wavelength
        )
        target = np.fft.ifftshift(grating)
        target = np.fft.fft2(target)
        target = np.fft.fftshift(target)
        # plt.imshow(abs(target[0,:,:]))
        # plt.show()
        target_comb = np.hstack([target.real,target.imag])
        target_combTensor = torch.Tensor(target_comb).to(device)
        optimizer.zero_grad()
        grating = torch.Tensor(grating).to(device)
        outputs = net(grating)
        loss = loss_func(outputs,target_combTensor)
        loss.backward()
        optimizer.step()
        print("epoch_{}: {}".format(epoch,loss.item()))
        epoches.append(epoch)
        loss_value.append(loss.item())
        if loss < 0.5 and epoch > 3000:
            break
    plt.plot(epoches,loss_value)
    plt.show()
    #test
    x_test = np.arange(-N // 2, N // 2, 1)
    X_test, Y_test = np.meshgrid(x_test, x_test)
    wavelength = 20
    angle = 60
    grating = np.sin(
        2 * np.pi * (X_test * np.cos(angle) + Y_test * np.sin(angle)) / wavelength
    )

    target_test = np.fft.ifftshift(grating)
    target_test = np.fft.fft2(target_test)
    target_test = np.fft.fftshift(target_test)
    grating = torch.Tensor(grating).to(device)
    test_output = net(grating)
    test_output_real = test_output[:,0:N]
    test_output_imag = test_output[:,N:2*N]
    test_fft = test_output_real + 1j * test_output_imag
    ift = np.fft.ifftshift(test_fft.cpu().detach().numpy())
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    ift = ift.real
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(grating.cpu().numpy(), cmap="gray")
    plt.subplot(222)
    plt.title("Predict FFT")
    plt.imshow(np.log(abs(test_fft.cpu().detach().numpy())), cmap="gray")
    plt.subplot(223)
    plt.title("True FFT")
    plt.imshow(np.log(abs(target_test)), cmap="gray")
    plt.subplot(224)
    plt.title("Restored image")
    plt.imshow(ift, cmap="gray")
    plt.tight_layout()
    plt.show()
    ift = torch.Tensor(ift).to(device)
    loss_test = loss_func(grating, ift)
    print(loss_test.item())





# def cal_phase(x):
# # yf_new  = np.zeros(2000,dtype=complex)
# #     threshold = max(abs(fft_results))/10
# #     for i in range(len(yf_new)):
# #         if abs(fft_results[i]) > threshold:
# #             yf_new[i] = fft_results[i] + yf_new[i]
# #
#     phase = np.zeros([len(x),len(x)])
#     for i in range(len(phase)):
#         for j in range(len(phase[i,:])):
#             ratio = x[i,j].imag / x[i,j].real
#             phase[i,j] = math.atan(ratio) * 180/np.pi
#     return phase
# def restore_FFT(Amplitude,phase):
#     fft = np.zeros([len(Amplitude), len(Amplitude)],dtype=complex)
#     for i in range(len(phase)):
#         for j in range(len(phase[i,:])):
#             fft[i,j] = Amplitude[i,j] * np.exp(1j*(phase[i,j] *np.pi/180)) + fft[i,j]
#     return fft

