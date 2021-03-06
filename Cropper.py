import numpy as np

import cv2

import os

import pickle

import matplotlib.pyplot as plt



class Montage(object):


    def __init__(self, min_size=100):


        self.size = 28


        self.min_size = min_size


        self.data_path = './data.pkl'


        self.data_dict = self.read_data()


        self.data = self.convert_data(self.data_dict)


    def resize_image(self, image):


        image_shape = image.shape


        size_min = np.min(image_shape[:2])

        size_max = np.max(image_shape[:2])


        scale = float(self.min_size) / float(size_max)


        image = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale)


        return image


    def convert_data(self, data):


        new_data = []


        for value in data:

            new_data.append(value['value'])


        return np.array(new_data)


    def read_data(self, data_path='./image'):


        if not os.path.exists(self.data_path):


            dir_list = os.listdir(data_path)


            data = []


            for dir_path in dir_list:


                file_type = os.path.splitext(dir_path)[1]

                if file_type != '.jpeg':

                    continue


                image_path = os.path.join(data_path, dir_path)

                image = cv2.imread(image_path)


                B, G, R = cv2.split(image)

                B = np.mean(B)

                G = np.mean(G)

                R = np.mean(R)

                mean_value = (B, G, R)


                value = {'path': image_path, 'value': mean_value}


                data.append(value)


            with open(self.data_path, 'wb') as f:

                pickle.dump(data, f)


            return data

        # else


        with open(self.data_path, 'rb') as f:

            data = pickle.load(f)


        return data


    def Splice_index(self, image_path):


        image = cv2.imread(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        image = self.resize_image(image)

        h, w = image.shape[:2]


        all_index = []


        for i in range(h):


            new_line = []


            for j in range(w):


                value = image[i, j, :]


                res = self.data - value

                res = np.sum(np.abs(res), axis=-1)


                res_index = np.argmin(res)


                new_line.append(res_index)


            all_index.append(new_line)


        return all_index, h, w


    def Splice(self, image_path):


        all_index, h, w = self.Splice_index(image_path)


        row = []

        for i in range(h):


            line = []

            line_index = all_index[i]


            for j in range(w):


                index = line_index[j]

                name = self.data_dict[index]['path']

                line.append(cv2.imread(name))


            line = np.hstack(line)

            row.append(line)


        image = np.vstack(row)

        return image


if __name__ == "__main__":

    montage = Montage(min_size=150) # min_size越大，精度越高；但相应图片处理时间越长
    
    image = montage.Splice('./0.jpg') # 传入图片路径

    plt.imshow(image)
    plt.show()

    image = montage.Splice('./0.png') # 传入图片路径

    plt.imshow(image)
    plt.show()
