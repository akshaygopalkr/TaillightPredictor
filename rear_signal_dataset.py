from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import torch
import os
import pdb


class RearSignalDataset(Dataset):

    def get_image_list(self):

        footage_name_list = [footage_name for footage_name in os.listdir('.\\rear_signal_dataset') if
                             'txt' not in footage_name]

        frame_directories = []

        # Go through each directory in the rear_signal dataset file
        for footage_name_list in footage_name_list:

            for footage_class in os.listdir('.\\rear_signal_dataset\\' + footage_name_list):

                curr_dir = '.\\rear_signal_dataset\\' + footage_name_list + '\\' + footage_class

                for frame_direcs in os.listdir(curr_dir):
                    curr_dir_2 = curr_dir + '\\' + frame_direcs + '\\light_mask\\'

                    # Get the images for this sequence
                    images = os.listdir(curr_dir_2)

                    paths = [curr_dir_2 + image_id for image_id in images]
                    frame_directories.append(paths)

        return frame_directories

    def __init__(self, transform=None):

        self.video_list = self.get_image_list()
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        # Get the image path for the idx
        vid_paths = self.video_list[idx]

        image_list = []


        label_str = vid_paths[0].split('\\')[-4][-3:]
        break_state = 1 if label_str[0] == 'B' else 0
        left_state = 1 if label_str[1] == 'L' else 0
        right_state = 1 if label_str[2] == 'R' else 0
        label = (break_state, left_state, right_state)

        print(label, label_str)

        for vid_path in vid_paths:

            image = cv2.imread(vid_path)
            image = torch.tensor(image)
            image = torch.reshape(image, (3, image.size()[0], image.size()[1]))

            if self.transform:
                image = self.transform(image)

            image_list.append(image)

        return image_list, label


if __name__ == '__main__':

    rear_signal_dataset = RearSignalDataset()

    for i in range(len(rear_signal_dataset)):

        image_list, label = rear_signal_dataset[i]
