from models import *
from torch.autograd import Variable
from tqdm import tqdm
from dataset import *
from torch.utils.data import DataLoader
import torch
from ignite.metrics import FID, InceptionScore
import PIL.Image as Image
from ignite.engine import Engine


device = "cuda:0"
net = Combine(256).to(device)
description = "Combine_shoes"

test_dataset = Shoes("./archive/train",
                   "./archive/train",
                   mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
path_checkpoint = './checkpoints/{}'.format(description)
checkpoint = torch.load(path_checkpoint)
net.load_state_dict(checkpoint['model_state_dict'])
fid_metric = FID(device=device)
is_metric = InceptionScore(device=device,output_transform=lambda x:x[0])

def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

def to_img(x):
    """
        Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor
    """

    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 256, 256)

    return x

def test(engine,batch):
    net.eval()
    for idx, data in enumerate(tqdm(test_dataloader)):
        uw_img, cl_img = data
        uw_img = Variable(uw_img).to(device)
        cl_img = Variable(cl_img, requires_grad=False).to(device)
        u_out,_ = net(uw_img)
        u_out = to_img(u_out)
        fake = interpolate(u_out)
        real = interpolate(cl_img)
        return fake,real
evaluator = Engine(test)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")

fid_values = []
is_values = []


evaluator.run(test_dataloader,max_epochs=1)
metrics = evaluator.state.metrics
fid_score = metrics['fid']
is_score = metrics['is']
fid_values.append(fid_score)
is_values.append(is_score)
print(f"*   FID : {fid_score:4f}")