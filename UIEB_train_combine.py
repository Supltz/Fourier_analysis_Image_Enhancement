from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import os
from dataset import *
from tqdm import tqdm
import numpy as np
from models import *
import argparse
import datetime
from tensorboardX import SummaryWriter
from MS_SSIM_L1_3 import MS_SSIM_L1_LOSS
import warnings
import torch.optim as optim
from focal_frequency_loss import FocalFrequencyLoss as FFL

#args and settings
warnings.filterwarnings("ignore")
now = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
parser = argparse.ArgumentParser('parameters', add_help=False)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--description', default="Combine", type=str)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--device', default="cuda:0", type=str)
parser.add_argument('--num_epoch', default=160, type=int)
parser.add_argument('--train_size', default=30000, type=int)
parser.add_argument('--val_size', default=3000, type=int)
parser.add_argument('--test_size', default=3000, type=int)
parser.add_argument('--data_path', default="./raw-890/raw-890", type=str)
parser.add_argument('--label_path', default="./UIEB-ref/",type=str)
parser.add_argument('--dataset', default="UIEB", type=str)
parser.add_argument('--FirstTimeRunning', default=None,required=True, type=str, help="No means reading from the checkpoint_file")
parser.add_argument('--model', default=None)
args = parser.parse_args()

def to_img(x):
    """
        Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor
    """

    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)

    return x
def evaluation_metric(outputs,cl_img,mse_scores,ssim_scores,psnr_scores):
    # calculating evaluation metric

    criterion_MSE = nn.MSELoss().to(device)
    mse_scores.append(criterion_MSE(outputs, cl_img).item())
    i = 0
    while(i < outputs.shape[0]):
        outputTensor = outputs[i,:,:,:]
        outputImage = (outputTensor * 255).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        clTensor = cl_img[i,:,:,:]
        clImage = (clTensor * 255).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        ssim_scores.append(ssim(outputImage, clImage, multichannel=True))
        psnr_scores.append(psnr(outputImage, clImage))
        i = i + 1

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss_fre = 0
    train_loss_img = 0
    mse_scores = []
    ssim_scores = []
    psnr_scores = []
    lambda1 = 4
    for idx, data in enumerate(tqdm(train_dataloader)):
        uw_img, cl_img = data
        uw_img = Variable(uw_img).to(device)
        cl_img = Variable(cl_img, requires_grad=False).to(device)
        optimizer.zero_grad()
        targets = np.fft.ifftshift(cl_img.cpu().numpy())
        targets = np.fft.fft2(targets)
        targets = np.fft.fftshift(targets)
        target_comb = np.concatenate([targets.real, targets.imag], axis=3)
        target_combTensor = torch.Tensor(target_comb).to(device)
        u_out, output = net(uw_img)
        u_out = to_img(u_out)
        loss_fre =  MSE(output, target_combTensor) / (N*N)
        loss_img = SSIM(u_out, cl_img)
        loss_focal = ffl(u_out, cl_img)
        loss = lambda1 * loss_img + loss_fre + loss_focal
        loss.backward()
        optimizer.step()
        train_loss_fre += loss_focal
        train_loss_img += loss_img
        #Evaluation metric
        evaluation_metric(u_out,cl_img,mse_scores,ssim_scores,psnr_scores)
    # store the result
    ssim_score = sum(ssim_scores) / (len(train_dataloader) * batch_size)
    psnr_score = sum(psnr_scores) / (len(train_dataloader) * batch_size)
    mse_score = sum(mse_scores) / (len(train_dataloader) * batch_size)
    tra_loss_img= train_loss_img /(len(train_dataloader) * batch_size)
    tra_loss_fre = train_loss_fre / (len(train_dataloader) * batch_size)
    progress = "\tloss_img: {} \tloss_fre: {}\tSSIM: {}\tPSNR: {}\tMSE: {}"\
        .format(tra_loss_img, tra_loss_fre, ssim_score,psnr_score,mse_score)
    print(progress)
    return tra_loss_img, tra_loss_fre,ssim_score,psnr_score,mse_score


@torch.no_grad()
def val(epoch):
    print('\nEpoch(validation): %d' % epoch)
    net.eval()
    val_loss_fre = 0
    val_loss_img = 0
    mse_scores = []
    ssim_scores = []
    psnr_scores = []
    for idx, data in enumerate(tqdm(val_dataloader)):
        uw_img, cl_img = data
        uw_img = Variable(uw_img).to(device)
        cl_img = Variable(cl_img, requires_grad=False).to(device)
        targets = np.fft.ifftshift(cl_img.cpu().numpy())
        targets = np.fft.fft2(targets)
        targets = np.fft.fftshift(targets)
        tar_abs = np.log(abs(targets))
        tar_abs = torch.from_numpy(tar_abs)
        uw_fft = np.fft.ifftshift(uw_img.cpu().numpy())
        uw_fft = np.fft.fft2(uw_fft)
        uw_fft = np.fft.fftshift(uw_fft)
        uw_abs = np.log(abs(uw_fft))
        uw_abs = torch.from_numpy(uw_abs)
        u_out, output = net(uw_img)
        u_out = to_img(u_out)
        loss_fre = ffl(u_out, cl_img)
        loss_img = SSIM(u_out, cl_img)
        output_real = output[:, :, :, 0:N]
        output_imag = output[:, :, :, N:2 * N]
        output_fft = output_real + 1j * output_imag
        output_abs = np.log(abs(output_fft.cpu().numpy()))
        output_abs = torch.from_numpy(output_abs)
        val_loss_fre += loss_fre
        val_loss_img += loss_img
        # Evaluation metric
        evaluation_metric(u_out, cl_img, mse_scores, ssim_scores, psnr_scores)
    # store the result
    if epoch % 4 == 0:
        save_image(uw_img.cpu().data, './results_combine/uw_img_{}.png'.format(epoch))
        save_image(cl_img.cpu().data, './results_combine/gt_img_{}.png'.format(epoch))
        save_image(u_out, './results_combine/outputs_img{}.png'.format(epoch))
        save_image(tar_abs, './results_combine/gt_fft{}.png'.format(epoch))
        save_image(output_abs, './results_combine/out_fft{}.png'.format(epoch))
        save_image(uw_abs, './results_combine/uw_fft{}.png'.format(epoch))
    ssim_score = sum(ssim_scores) / (len(val_dataloader) * batch_size)
    psnr_score = sum(psnr_scores) / (len(val_dataloader) * batch_size)
    mse_score = sum(mse_scores) / (len(val_dataloader) * batch_size)
    val_loss_img = val_loss_img /(len(val_dataloader) * batch_size)
    val_loss_fre = val_loss_fre / (len(val_dataloader) * batch_size)
    progress = "\tloss_img: {}\tloss_fre: {}\tSSIM: {}\tPSNR: {}\tMSE: {}"\
        .format(val_loss_img, val_loss_fre, ssim_score,psnr_score,mse_score)
    print(progress)
    return val_loss_img, val_loss_fre, ssim_score,psnr_score,mse_score


def main():

    for epoch in range(start_epoch + 1, num_epoch):
        loss_dict = {}
        ssim_dict = {}
        psnr_dict = {}
        mse_dict = {}
        tra_loss_img, tra_loss_fre,tra_ssim,tra_psnr,tra_mse = train(epoch)
        val_loss_img, val_loss_fre, val_ssim, val_psnr, val_mse = val(epoch)
        loss_dict.update({'train_loss_img': tra_loss_img})
        loss_dict.update({'train_loss_fre': tra_loss_fre})
        loss_dict.update({'val_loss_img': val_loss_img})
        loss_dict.update({'val_loss_fre': val_loss_fre})
        ssim_dict.update({'train_ssim': tra_ssim})
        psnr_dict.update({'train_psnr': tra_psnr})
        mse_dict.update({'train_mse': tra_mse})

        ssim_dict.update({'val_ssim': val_ssim})
        psnr_dict.update({'val_psnr': val_psnr})
        mse_dict.update({'val_mse': val_mse})
        writer1 = SummaryWriter("run-{}-{}".format(description, now))
        writer1.add_scalars('loss', loss_dict, global_step=epoch)
        writer1.add_scalars('ssim', ssim_dict, global_step=epoch)
        writer1.add_scalars('psnr', psnr_dict, global_step=epoch)
        writer1.add_scalars('mse', mse_dict, global_step=epoch)
        writer1.close()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, './checkpoints/{}'.format(description))
# 回溯的时候 时间不一样







if __name__ == "__main__":

    # set parameters
    data_path = args.data_path
    label_path = args.label_path
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    batch_size = args.batch_size
    yes_no = args.FirstTimeRunning
    num_epoch = args.num_epoch
    start_epoch = 0
    description = args.description

    #make paths
    if not os.path.exists('./results_combine'):
        os.mkdir('./results_combine')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')



    # Define datasets and dataloaders
    print('==> Preparing data...')
    train_dataset = UIEB(data_path,
                                 label_path,
                                 mode='train')

    val_dataset = UIEB(data_path,
                               label_path,
                               mode='val')

    # test_dataset = UIEB(data_path,
    #                             label_path,
    #                             size=test_size,
    #                             test_start=46000,
    #                             mode='test')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True,drop_last=True)

    #Device
    device = args.device

    # Define modelsand optimizers
    print("start the nets")
    N = 256
    net = Combine(N)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    SSIM = MS_SSIM_L1_LOSS().to(device)
    MSE = nn.MSELoss().to(device)
    ffl = FFL(loss_weight=1.0, alpha=1.0).to(device)



    #Recover
    if (yes_no == "no"):
        print('-----------------------------')
        path_checkpoint = './checkpoints/{}'.format(description)
        checkpoint = torch.load(path_checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print("start_epoch:", start_epoch + 1)
        print('-----------------------------')
        main()
    if (yes_no == "yes"):
        main()
