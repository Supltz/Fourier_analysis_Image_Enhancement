from skimage import color, filters
import math
from dataset import *
from torch.utils.data import DataLoader
from models import *
from tqdm import tqdm
from torch.autograd import Variable
def to_img(x):
    """
        Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor
    """

    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)

    return x
def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    l = lab[:, :, 0]

    # 1st term
    chroma = (lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2) ** 0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc) ** 2)) ** 0.5

    # 2nd term
    top = np.int32(0.01 * l.shape[0] * l.shape[1])
    sl = np.sort(l, axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[::top]) - np.mean(sl[::top])

    # 3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0:
            satur.append(0)
        elif l1[i] == 0:
            satur.append(0)
        else:
            satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.5
    p2 = 0.4
    p3 = 0.1

    # 1st term UICM
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]
    rgl = np.sort(rg, axis=None)
    ybl = np.sort(yb, axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = np.int32(al1 * len(rgl))
    T2 = np.int32(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr - uyb) ** 2)

    uicm = -0.0268 * np.sqrt(urg ** 2 + uyb ** 2) + 0.1586 * np.sqrt(s2rg + s2yb)

    # 2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:, :, 0] * filters.sobel(rgb[:, :, 0])
    Gsobel = rgb[:, :, 1] * filters.sobel(rgb[:, :, 1])
    Bsobel = rgb[:, :, 2] * filters.sobel(rgb[:, :, 2])

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    # 3rd term UIConM
    logameeR = logamee(rgb[:, :, 0])
    logameeG = logamee(rgb[:, :, 1])
    logameeB = logamee(rgb[:, :, 2])

    uiconm = (logameeR + logameeG + logameeB) / 3

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm

    return uiqm, uciqe


def eme(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]

            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            if blockmin == 0.0:
                eme += 0
            elif blockmax == 0.0:
                eme += 0
            else:
                eme += w * math.log(blockmax / blockmin)
    return eme


def plipsum(i, j, gamma=1026):
    return i + j - i * j / gamma


def plipsub(i, j, k=1026):
    return k * (i - j) / (k - j)


def plipmult(c, j, gamma=1026):
    return gamma - gamma * (1 - j / gamma) ** c


def logamee(ch, blocksize=8):
    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)

    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i + 1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j + 1) * blocksize
            else:
                yrb = ch.shape[1]

            block = ch[xlb:xrb, ylb:yrb]
            blockmin = np.float64(np.min(block))
            blockmax = np.float64(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            if bottom == 0.0:
                s += 0
            elif top == 0.0:
                s += 0
            else:
                s += (top / bottom) * np.log(top / bottom)

    return plipmult(w, s)


def main():


    device = "cuda:0"
    net = Combine(256).to(device)
    name = "Combine"
    path_checkpoint = './checkpoints/{}'.format(name)
    checkpoint = torch.load(path_checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    test_dataset = UIEB("./raw-890/raw-890",
                        "./UIEB-ref/",
                        mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    sumuiqm, sumuciqe = 0., 0.
    net.eval()
    for idx, data in enumerate(tqdm(test_dataloader)):
        uw_img, cl_img = data
        uw_img = Variable(uw_img).to(device)
        u_out,_ = net(uw_img)
        u_out = to_img(u_out)
        corrected = (u_out.squeeze(dim=0) * 255).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        uiqm, uciqe = nmetrics(corrected)
        sumuiqm += uiqm
        sumuciqe += uciqe
        # with open(os.path.join("./UIEB_test_results", 'metrics_{}.txt'.format(name)), 'a') as f:
        #     f.write('uiqm={} uciqe={}\n'.format(uiqm, uciqe))

    muiqm = sumuiqm / 30
    muciqe = sumuciqe / 30

    with open(os.path.join("./UIEB_test_results", 'metrics_{}.txt'.format(name)), 'a') as f:
        f.write('Average: uiqm={} uciqe={}\n'.format(muiqm, muciqe))


if __name__ == '__main__':
    main()