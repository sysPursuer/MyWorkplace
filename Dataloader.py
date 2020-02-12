import torch
from PIL import Image
import torch.utils.data as Data
from torchvision import transforms
import os
from torchvision.datasets import ImageFolder

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class MyDataset(Data.Dataset):
    def __init__(self,A_path,B_path,max_dataset_size):
        super(MyDataset,self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.A_paths = self.get_imgPaths(A_path,max_dataset_size)
        self.B_paths = self.get_imgPaths(B_path,max_dataset_size)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def get_imgPaths(self,dir, max_dataset_size):
        '''return the img_paths in the dir'''
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]

    def __getitem__(self,index):
        in_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
        ])
        
        A_path = self.A_paths[index%self.A_size]
        A_img = Image.open(A_path).convert('RGB')
        A_img = in_transform(A_img)
        B_path = self.B_paths[index%self.B_size]
        B_img = in_transform(Image.open(B_path).convert('RGB'))
        
        return A_img.to(self.device),B_img.to(self.device)

    def __len__(self):
        return max(self.A_size,self.B_size)

def get_dataloader(dir1,dir2,max_size=100):
    dataset = MyDataset(dir1,dir2,max_size)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=True
    )
    return loader
if __name__ == '__main__':
    loader = get_dataloader('./horse2zebra/trainA/','./images/')
    for step,(real_A,real_B) in enumerate(loader):
        print('Epoch: ', 0, '| Step: ', step, '| real_A: ',
              real_A.size(),'| real_B',real_B.size())