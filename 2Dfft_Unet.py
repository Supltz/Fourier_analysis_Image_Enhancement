import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models import *
from dataset import *
from torch.utils.data import DataLoader
import pickle
import cv2
from focal_frequency_loss import FocalFrequencyLoss as FFL



N = 256
model = "unet" #unet linear
datatype = "cifar10"#noise cifar10
batchsize = 8
device = 'cuda:0'
MSE = nn.MSELoss().to(device)
L1 = nn.L1Loss().to(device)
ffl = FFL(loss_weight=1.0,alpha=1.0).to(device)



if model=="linear" and datatype=="cifar10":
    net = LinearNet(N)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    train_dataset = UWdataset("./label",mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
    shuffle=True,num_workers=0,pin_memory=True,drop_last=True)
    for epoch in range(4000):
        net.train()
        loss_sum = 0
        loss_sum_unsup  = 0
        for idx, data in enumerate(tqdm(train_dataloader)):
            loss_batch = 0
            loss_unsup_batch = 0
            for i in range(batchsize):
                data = data.to(device)
                optimizer.zero_grad()
                target_np = np.fft.ifftshift(data[i, :, :, :].squeeze().cpu().numpy())
                target_np = np.fft.fft2(target_np)
                target_np = np.fft.fftshift(target_np)
                target_comb = np.hstack([target_np.real, target_np.imag])
                target_combTensor = torch.Tensor(target_comb).to(device)
                output = net(data.squeeze()[i,:,:])
                output_real = output[:, 0:N]
                output_imag = output[:, N:2 * N]
                output_fft = output_real + 1j * output_imag
                ift = np.fft.ifftshift(output_fft.cpu().detach().numpy())
                ift = np.fft.ifft2(ift)
                ift = np.fft.fftshift(ift)
                ift = ift.real
                ift = torch.Tensor(ift).to(device)
                loss_unsup = MSE(data[i, :, :, :].squeeze(),ift)
                loss = MSE(output,target_combTensor)
                loss.backward()
                optimizer.step()
                loss_batch += loss.item()
                loss_unsup_batch += loss_unsup.item()
            loss_sum += loss_batch
            loss_sum_unsup += loss_unsup_batch
        print("epoch_{}: {}".format(epoch, loss_sum/(len(train_dataloader) * batchsize)))
        print("epoch_{}: {}".format(epoch, loss_sum_unsup / (len(train_dataloader) * batchsize)))
        if epoch==30:
            break
    img = cv2.imread("./Test/44_img_.png",cv2.IMREAD_GRAYSCALE)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((180, 270)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor()
    ])
    img = transform(img)
    target_test = np.fft.ifftshift(img)
    target_test = np.fft.fft2(target_test)
    target_test = np.fft.fftshift(target_test)
    img_gray = torch.Tensor(img).to(device)
    output_test = net(img_gray.squeeze())
    test_output_real = output_test[:, 0:N]
    test_output_imag = output_test[:, N:2 * N]
    test_fft = test_output_real + 1j * test_output_imag
    ift_test = np.fft.ifftshift(test_fft.cpu().detach().numpy())
    ift_test = np.fft.ifft2(ift_test)
    ift_test = np.fft.fftshift(ift_test)
    ift_test = ift_test.real
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(img_gray.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(222)
    plt.title("Predict FFT")
    plt.imshow(np.log(abs(test_fft.cpu().detach().numpy())), cmap="gray")
    plt.subplot(223)
    plt.title("True FFT")
    plt.imshow(np.log(abs(target_test.squeeze())), cmap="gray")
    plt.subplot(224)
    plt.title("Restored image")
    plt.imshow(ift_test, cmap="gray")
    plt.tight_layout()
    plt.show()
    ift_test = torch.Tensor(ift_test).to(device)
    loss_test = MSE(img_gray, ift_test)
    print(loss_test.item())


elif model=="unet" and datatype=="cifar10":
    net = Combine(256)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_dataset = UWdataset("./label",mode="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batchsize,
                                  shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    for epoch in range(4000):
        net.train()
        loss_sum = []
        loss_unsup_sum = []
        lambda1 = 4
        for idx, data in enumerate(tqdm(train_dataloader)):
            #targets = torch.fft.fftshift(torch.fft.fftn(data)).to(device)
            targets = np.fft.ifftshift(data.cpu().numpy())
            targets = np.fft.fft2(targets)
            targets = np.fft.fftshift(targets)
            target_comb = np.concatenate([targets.real, targets.imag],axis=3)
            target_combTensor = torch.Tensor(target_comb).to(device)
            data = data.to(device)
            optimizer.zero_grad()
            u_out,outputs= net(data)
            # outputs_real = outputs[:,:,:,0:N]
            # outputs_imag = outputs[:,:,:,N:2 * N]
            #outputs = torch.complex(outputs_real,outputs_imag) 0.00016
            #ifft = torch.real(torch.fft.ifftn(torch.fft.ifftshift(outputs))).to(device)
            loss_unsup = MSE(u_out, data)
            loss = MSE(outputs, target_combTensor) / (N*N) + lambda1 * loss_unsup
            loss_sum.append(loss.item())
            loss_unsup_sum.append(loss_unsup.item())
            loss.backward()
            optimizer.step()
        print("epoch_{}: {}".format(epoch, sum(loss_sum) / len(loss_sum)))
        print("epoch_{}_restore_loss: {}".format(epoch, sum(loss_unsup_sum) / len(loss_unsup_sum)))
        if sum(loss_sum) / len(loss_sum) < 0.018:
            break
    #test
    img = cv2.imread("./Test/44_img_.png", cv2.IMREAD_GRAYSCALE)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((270, 360)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])
    img = transform(img)
    img_gray = torch.Tensor(img).to(device)
    target_test = np.fft.ifftshift(img)
    target_test = np.fft.fft2(target_test)
    target_test = np.fft.fftshift(target_test)
    img_gray = torch.Tensor(img).to(device)
    u_out_test, test_output = net(img_gray.unsqueeze(dim=0))
    test_output_real = test_output[:,:,:,0:N]
    test_output_imag = test_output[:,:,:,N:2 * N]
    test_fft = test_output_real + 1j * test_output_imag
    ift_test = np.fft.ifftshift(test_fft.cpu().detach().numpy())
    ift_test = np.fft.ifft2(ift_test)
    ift_test = np.fft.fftshift(ift_test)
    ift_test = ift_test.real
    ift_test = torch.Tensor(ift_test).to(device)
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(u_out_test[0,0,:,:].cpu().detach().numpy(), cmap="gray")
    plt.subplot(222)
    plt.title("Predict FFT")
    plt.imshow(np.log(abs(test_fft[0,0,:,:].cpu().detach().numpy())), cmap="gray")
    plt.subplot(223)
    plt.title("True FFT")
    plt.imshow(np.log(abs(target_test[0,:,:])), cmap="gray")
    plt.subplot(224)
    plt.title("Restored image")
    plt.imshow(ift_test[0,0,:,:].cpu().numpy(), cmap="gray")
    plt.tight_layout()
    plt.savefig("./results.png")
    plt.show()
    loss_test = MSE(img_gray, ift_test[0,:,:,:])
    print(loss_test.item())

elif model=="linear" and datatype=="noise":
    net = LinearNet(112)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(2500):
        noise = torch.rand(N, N).to(device)
        target = np.fft.ifftshift(noise.cpu())
        target = np.fft.fft2(target)
        target = np.fft.fftshift(target)
        target_comb = np.hstack([target.real, target.imag])
        target_combTensor = torch.Tensor(target_comb).to(device)
        optimizer.zero_grad()
        outputs = net(noise)
        loss = MSE(outputs, target_combTensor)
        loss.backward()
        optimizer.step()
        print("epoch_{}: {}".format(epoch, loss.item()))
        img = cv2.imread("./Test/44_img_.png", cv2.IMREAD_GRAYSCALE)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((180, 270)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor()
        ])
        img = transform(img)
        target_test = np.fft.ifftshift(img)
        target_test = np.fft.fft2(target_test)
        target_test = np.fft.fftshift(target_test)
        img_gray = torch.Tensor(img).to(device)
        output_test = net(img_gray.squeeze())
        test_output_real = output_test[:, 0:N]
        test_output_imag = output_test[:, N:2 * N]
        test_fft = test_output_real + 1j * test_output_imag
        ift_test = np.fft.ifftshift(test_fft.cpu().detach().numpy())
        ift_test = np.fft.ifft2(ift_test)
        ift_test = np.fft.fftshift(ift_test)
        ift_test = ift_test.real
        plt.subplot(221)
        plt.title("Original image")
        plt.imshow(img_gray.squeeze().cpu().numpy(), cmap="gray")
        plt.subplot(222)
        plt.title("Predict FFT")
        plt.imshow(np.log(abs(test_fft.cpu().detach().numpy())), cmap="gray")
        plt.subplot(223)
        plt.title("True FFT")
        plt.imshow(np.log(abs(target_test.squeeze())), cmap="gray")
        plt.subplot(224)
        plt.title("Restored image")
        plt.imshow(ift_test, cmap="gray")
        plt.tight_layout()
        plt.savefig("./results_2D/{}.jpg".format(epoch))
        #plt.show()
        # test
    img = cv2.imread("./Test/44_img_.png", cv2.IMREAD_GRAYSCALE)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((180, 270)),
        transforms.CenterCrop((112, 112)),
        transforms.ToTensor()
    ])
    img = transform(img)
    target_test = np.fft.ifftshift(img)
    target_test = np.fft.fft2(target_test)
    target_test = np.fft.fftshift(target_test)
    img_gray = torch.Tensor(img).to(device)
    output_test = net(img_gray.squeeze())
    test_output_real = output_test[:, 0:N]
    test_output_imag = output_test[:, N:2 * N]
    test_fft = test_output_real + 1j * test_output_imag
    ift_test = np.fft.ifftshift(test_fft.cpu().detach().numpy())
    ift_test = np.fft.ifft2(ift_test)
    ift_test = np.fft.fftshift(ift_test)
    ift_test = ift_test.real
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(img_gray.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(222)
    plt.title("Predict FFT")
    plt.imshow(np.log(abs(test_fft.cpu().detach().numpy())), cmap="gray")
    plt.subplot(223)
    plt.title("True FFT")
    plt.imshow(np.log(abs(target_test.squeeze())), cmap="gray")
    plt.subplot(224)
    plt.title("Restored image")
    plt.imshow(ift_test, cmap="gray")
    plt.tight_layout()
    plt.show()
    ift_test = torch.Tensor(ift_test).to(device)
    loss_test = MSE(img_gray, ift_test)
    print(loss_test.item())

elif model=="unet" and datatype=="noise":
    net = UNet(2,1,2)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    for epoch in range(8000):
        noise = torch.rand(1,1,N, N).to(device)
        target = np.fft.ifftshift(noise.squeeze(dim=0).squeeze(dim=1).cpu())
        target = np.fft.fft2(target)
        target = np.fft.fftshift(target)
        target_real = torch.Tensor(target.real).to(device)
        target_imag = torch.Tensor(target.imag).to(device)
        optimizer.zero_grad()
        output_real,output_imag = net(noise)
        output = output_real + 1j * output_imag
        loss = MSE(output_real,target_real) + MSE(output_imag,target_imag)
        loss.backward()
        optimizer.step()
        print("epoch_{}: {}".format(epoch, loss.item()))
        if loss.item() < 80:
            break
 #test
    with open("./cifar_10/test_batch", 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    image = np.reshape(dict[b'data'][1, :], [3, 32, 32])
    img = np.moveaxis(image, 0, -1)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    target_test = np.fft.ifftshift(img_gray)
    target_test = np.fft.fft2(target_test)
    target_test = np.fft.fftshift(target_test)
    img_gray = torch.Tensor(img_gray).to(device)
    img_gray = img_gray.unsqueeze(dim=0).unsqueeze(dim=0)
    test_output_real,test_output_imag = net(img_gray)
    test_fft = test_output_real + 1j * test_output_imag
    ift_test = np.fft.ifftshift(test_fft.cpu().detach().numpy())
    ift_test = np.fft.ifft2(ift_test)
    ift_test = np.fft.fftshift(ift_test)
    ift_test = ift_test.real
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(img_gray[0,0,:,:].cpu().numpy(), cmap="gray")
    plt.subplot(222)
    plt.title("Predict FFT")
    plt.imshow(np.log(abs(test_fft[0,0,:,:].cpu().detach().numpy())), cmap="gray")
    plt.subplot(223)
    plt.title("True FFT")
    plt.imshow(np.log(abs(target_test)), cmap="gray")
    plt.subplot(224)
    plt.title("Restored image")
    plt.imshow(ift_test[0,0,:,:], cmap="gray")
    plt.tight_layout()
    plt.show()
    ift_test = torch.Tensor(ift_test).to(device)
    loss_test = MSE(img_gray, ift_test)
    print(loss_test.item())



