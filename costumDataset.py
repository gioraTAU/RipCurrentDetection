from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image, ImageDraw


def draw_rect(image, box):
    """Present image with bounding box on top

    Parameters
    ----------
    image : numpy.ndarray
        numpy image

    box: numpy.ndarray
        Numpy array containing bounding boxes of shape `1 X 4` and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    Returns
    -------

    open new figure with image and bounding box

    """
    box = box[0]
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    img1 = ImageDraw.Draw(image)
    shape = [(x1, y1), (x2, y2)]
    img1.rectangle(shape, outline="red", width=4)
    image.show()


class RipCurrentDataset(Dataset):
    """ Rip current detector dataset. """

    def __init__(self, dframe, image_dir, transform=None):
        """

        :param dframe: Dataframe object of csv file "aug_data_label.csv" generated by fix_size_and_aug
        :param image_dir: path where all fixed size and augment images are saved
        :param transform: the transform to be operated converting PIL image to torch tensor
        """
        super().__init__()

        self.df = dframe
        self.images_ids = self.df['Name'].unique()
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return self.images_ids.shape[0]

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item : int
            id number to get one image from the dataset

        Returns
        -------

        img_tensor: torch.tensor
            Image as torch tensor object ready to be inserted into the Deep Neural Network

        target: dictionary
            Python dictionary contains image id, bounding box location (x1, y1, x2, y2) and label 0 - no rip, 1 - rip

        """
        img_name = self.images_ids[item]
        img_data = self.df[self.df['Name'] == img_name]

        img = Image.open(self.image_dir + img_name).convert("RGB")
        img_tensor = self.transform(img)

        x1, y1, x2, y2 = torch.tensor(img_data['x1'].values, dtype=torch.int64), torch.tensor(img_data['y1'].values, dtype=torch.int64), \
                         torch.tensor(img_data['x2'].values, dtype=torch.int64), torch.tensor(img_data['y2'].values, dtype=torch.int64)
        label = torch.tensor(img_data['label'].values, dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor(item)

        if label == 1:
            target['box'] = torch.cat((x1.unsqueeze(0), y1.unsqueeze(0), x2.unsqueeze(0), y2.unsqueeze(0)), dim=1)[0]
        else:
            target['box'] = torch.zeros((0, 4), dtype=torch.int64)

        target['labels'] = label[0]

        return img_tensor, target


if __name__ == '__main__':

    df = pd.read_csv(r'..\Data\aug_data_labels.csv')
    img_dir = r'..\Data\fixed_data\\'
    trans = T.ToTensor()
    train_ds = RipCurrentDataset(df, img_dir, trans)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    a = next(iter(train_dl))

    print('stam')
