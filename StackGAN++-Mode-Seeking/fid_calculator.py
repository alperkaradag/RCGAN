import os
from model import G_NET, FID_INCEPTION, INCEPTION_V3
from datasets import TextDataset
import torchvision.transforms as transforms
import torch
from trainer import weights_init
from torch.autograd import Variable
import numpy as np
from trainer import imread, get_activations, frechet_distance, get_fid_stats, compute_inception_score, negative_log_posterior_probability
from tqdm import tqdm
from miscc.utils import mkdir_p
from PIL import Image


def save_singleimages(images, save_dir, name):
    for i in range(len(images)):
        folder = save_dir
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)

        fullpath = os.path.join(save_dir, name) + ".png"
        # range from [-1, 1] to [0, 255]
        img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
        ndarr = img.permute(1, 2, 0).data.cpu().numpy()
        im = Image.fromarray(ndarr)
        im.save(fullpath)

DATA_DIR= 'C:\\Users\\Alper\\PycharmProjects\\MSGAN\\datasets\\birds'
imsize = 256
image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
dataset = TextDataset(DATA_DIR, split='test', base_size=64,
                              transform=image_transform)

nz = 100
n_samples = 10

fid_model = FID_INCEPTION()
fid_model.cuda()
fid_model.eval()

inception_model = INCEPTION_V3()
inception_model.cuda()
inception_model.eval()

G_NET_Path = 'C:\\Users\\alper\\PycharmProjects\\MSGAN\\StackGAN++-Mode-Seeking\\models\\their.pth'
netG = G_NET()
netG.apply(weights_init)
torch.cuda.set_device(0)
netG = netG.cuda()
netG = torch.nn.DataParallel(netG, device_ids=[0])
state_dict = \
    torch.load(G_NET_Path,
               map_location=lambda storage, loc: storage)
netG.load_state_dict(state_dict)


noise = Variable(torch.FloatTensor(n_samples, nz))
noise = noise.cuda()
netG.eval()

def generate_fake_images():
    emb = dataset.embeddings
    emb = np.reshape(emb, (emb.shape[0]*emb.shape[1], emb.shape[2]))
    np.random.shuffle(emb)
    fake_img_list = []
    print("Generating fake images")
    save_dir = "D:\\results\\fid"
    predictions = []
    inception_score_list = []
    for i in tqdm(range(200)):
        t_embeddings = emb[i]
        t_embeddings = torch.from_numpy(t_embeddings)
        t_embeddings = Variable(t_embeddings).cuda()
        noise.data.normal_(0, 1)
        for j in range(n_samples):
            noise_j = noise[j].unsqueeze(0)
            t_embeddings = t_embeddings.view(1, -1)
            fake_imgs, _, _ = netG(noise_j, t_embeddings)
            save_singleimages(fake_imgs[-1].data.cpu(), save_dir, "_"+str(i*n_samples+j))
            fake_img_list.append(fake_imgs[-1].data.cpu())

            pred = inception_model(fake_imgs[-1].detach())
            pred = pred.data.cpu().numpy()
            predictions.append(pred)
    predictions = np.concatenate(predictions, 0)
    mean, std = compute_inception_score(predictions, 10)
    mean_nlpp, std_nlpp = \
        negative_log_posterior_probability(predictions, 10)
    inception_score_list.append((mean, std, mean_nlpp, std_nlpp))
    # print(fake_img_list[0].shape)

    return [os.path.join(save_dir, i) for i in os.listdir(save_dir)], np.mean(inception_score_list)

def get_real_images():
    imgs_dir = os.path.join(DATA_DIR, "CUB_200_2011", "images")
    total_imgs = []
    print("Loading real images")
    for i in os.listdir(imgs_dir):
        if os.path.isdir(os.path.join(imgs_dir, i)):
            for j in os.listdir(os.path.join(imgs_dir, i)):
                total_imgs.append(os.path.join(i, j))
    idx = np.random.default_rng().choice(len(total_imgs), 2500, replace=False)
    selected_imgs = []
    for i in idx:
        selected_imgs.append(os.path.join(imgs_dir, total_imgs[i]))
    return selected_imgs


if __name__ == "__main__":
    lfid = []
    lis = []
    for i in range(5):
        fakes, inception_score = generate_fake_images()
        reals = get_real_images()
        act_real = get_activations(reals, fid_model)
        act_fake = get_activations(fakes, fid_model)
        mu_r, sigma_r = get_fid_stats(act_real)
        mu_f, sigma_f = get_fid_stats(act_fake)
        fid = frechet_distance(mu_r, sigma_r, mu_f, sigma_f)
        lfid.append(fid)
        lis.append(inception_score)
        print(i, fid, inception_score)
    # print("mu_real: {}, sigma_real: {}".format(mu_r, sigma_r))
    # print("mu_fake: {}, sigma_fake: {}".format(mu_f, sigma_f))
    print("means:", (np.mean(lfid), np.mean(lis)))
    with open("D:\\results\\fids.txt", "w+") as f:
        for i in list(zip(lfid, lis)):
            f.write("%f %f " % (i[0], i[1]))
        f.write("\n")
        f.write("%f %f " % (np.mean(lfid), np.mean(lis)))
