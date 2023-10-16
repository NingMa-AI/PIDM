import numpy as np
import os
import os.path
from PIL import Image


# def pil_loader(path):
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')

# def make_dataset_fromlist(image_list):
#     # with open(image_list) as f:
#     image_index = [x.split(' ')[0] for x in image_list]
#     # with open(image_list) as f:
#     label_list = []
#     selected_list = []
#     for ind, x in enumerate(image_list):
#         label = x.split(' ')[1].strip()
#         label_list.append(int(label))
#         selected_list.append(ind)
#     image_index = np.array(image_index)
#     label_list = np.array(label_list)
#     image_index = image_index[selected_list]
#     return image_index, label_list


# def return_classlist(image_list):
#     with open(image_list) as f:
#         label_list = []
#         for ind, x in enumerate(f.readlines()):
#             label = x.split(' ')[0].split('/')[-2]
#             if label not in label_list:
#                 label_list.append(str(label))
#     return label_list


class Text(object):
    def __init__(self, text, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        texts, labels = text[0],text[1]
        self.texts = texts
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        self.test = test
        self.return_index=False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        # path = os.path.join(self.root, self.imgs[index])
        # target = self.labels[index]
        # img = self.loader(path)
        single_text=self.texts[index]
        target=self.labels[index]
        # if self.transform is not None:
        #     single_text = self.transform(single_text)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            if self.return_index:
                return single_text, target,index
            else:
                return single_text, target
        else:
            return single_text, target, self.texts[index]

    def __len__(self):
        return len(self.texts)
