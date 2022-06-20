import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import glob

class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = []
        for file in glob.glob(val_data_dir + "haze/*"):
            name = file.split('/')[-1]
            if name not in val_list:
                val_list.append(name)
        haze_names = [i.strip() for i in val_list]
        gt_names = [i.split('_')[0] for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'haze/' + haze_name).convert('RGB')
        try: 
            gt_img = Image.open(self.val_data_dir + 'clear/' + gt_name + '.jpg').convert('RGB')
        except:
            gt_img = Image.open(self.val_data_dir + 'clear/' + gt_name + '.png').convert('RGB')
        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
