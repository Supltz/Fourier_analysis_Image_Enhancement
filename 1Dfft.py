"""
Train a neural network to implement the discrete Fourier transform
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import  math
from tqdm import tqdm

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    # 2pi because np.sin takes radians
    y = np.cos(2 * np.pi * freq * x + np.pi/4) + 2 * np.sin(5 * np.pi * freq * x + np.pi/4)
    return x, y

# def filter(fft_results):
#     yf_new  = np.zeros(2000,dtype=complex)
#     threshold = max(abs(fft_results))/10
#     for i in range(len(yf_new)):
#         if abs(fft_results[i]) > threshold:
#             yf_new[i] = fft_results[i] + yf_new[i]
#
#     phase = np.zeros(2000)
#     for i in range(len(phase)):
#         phase[i] = math.atan2(yf_new.imag[i],yf_new.real[i]) * 180/np.pi
#     return yf_new,phase

def create_fourier_weights(signal_length):
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
    return np.hstack([np.cos(theta_vals), -np.sin(theta_vals)])


# yf = np.fft.fft(y)
# y_filtered,_ = filter(yf)
# # y_re = np.fft.ifft(y_filtered)
# plt.plot(x,y_re)
# plt.show()
# plt.plot(yf.real, yf.imag,"g*")
# plt.show()
# plt.plot(xf, phase)
# plt.show()


#train on white noise
N = 400
fft_weight = create_fourier_weights(N)
loss_func = nn.MSELoss()
device = 'cuda:0'
layer_1 = nn.Linear(N,N*2,bias=False)
layer_1.to(device)
optimizer = optim.Adam(layer_1.parameters(),lr=0.01)
for epoch in tqdm(range(4800)):
    y = np.random.random([1, N]) - 0.5
    y_tensor = torch.Tensor(y).to(device)
    yf = np.fft.fft(y)
    yf_combine = np.hstack([yf.real,yf.imag])
    yf_combineTensor = torch.Tensor(yf_combine).to(device)
    fft_weight_tensor = torch.Tensor(fft_weight).to(device)
    optimizer.zero_grad()
    outputs = layer_1(y_tensor)
    loss = loss_func(outputs,yf_combineTensor).to(device)
    loss_weight = loss_func(layer_1.state_dict()['weight'], fft_weight_tensor.transpose(0, 1))
    loss.backward()
    optimizer.step()
    print("Loss_fft:{}\n".format(loss))
    print("Loss_Weight:{}\n".format(loss_weight))
# plt.subplot(2, 1, 1)
# plt.title("Train_weights")
# plt.imshow(layer_1.state_dict()['weight'].cpu().transpose(0,1).detach().numpy(), vmin=-1, vmax=1, cmap='coolwarm')
# plt.subplot(2, 1, 2)
# plt.title("Actual_weights")
# plt.imshow(fft_weight, vmin=-1, vmax=1,cmap='coolwarm')
# plt.tight_layout()
# #plt.savefig("./results/{}.jpg".format(epoch))
# plt.show()

#test it on a sine wave
SAMPLE_RATE = 200  # Hertz
DURATION = 2  # Seconds
freq = 5
N_sine = SAMPLE_RATE*DURATION #same with N
x_sine,y_sine = generate_sine_wave(freq,SAMPLE_RATE,DURATION)


y_sine_Tensor = torch.Tensor(y_sine).to(device)
outputs_sine = layer_1(y_sine_Tensor)
outputs_sine_real = outputs_sine[0:N]
outputs_sine_imag = outputs_sine[N:N*2]

# plt.plot(outputsreal.detach().numpy(), outputs_imag.detach().numpy(),"g*")
# plt.show()
res = outputs_sine_real + 1j*outputs_sine_imag
y_re = np.fft.ifft(res.cpu().detach().numpy())
y_true = np.fft.fft(y_sine)
plt.subplot(2,1,1)
plt.title("Original Amplitude Specturm")
xf = np.fft.fftfreq(N,1/SAMPLE_RATE)
plt.plot(xf, np.abs(y_true)/N)
plt.subplot(2,1,2)
plt.title("Predicted Amplitude Specturm")
xf = np.fft.fftfreq(N,1/SAMPLE_RATE)
plt.plot(xf, np.abs(res.cpu().detach().numpy()/N))
plt.tight_layout()
plt.show()

# plt.subplot(2,1,1)
# plt.title("Original Sinusoidal wave")
# plt.plot(x_sine,y_sine)
# plt.subplot(2,1,2)
# plt.title("Reconstructed Sinusoidal wave")
# plt.plot(x_sine, y_re)
# plt.tight_layout()
# plt.show()
#

