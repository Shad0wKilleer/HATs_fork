import torch


class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, batch_size):
        return_images = []
        for i in range(batch_size):
            tmp = self.images.pop(0)
            return_images.append(tmp.unsqueeze(0).clone())
        self.num_imgs = len(self.images)

        return_images = torch.cat(return_images, 0)
        return return_images

    def add(self, images):
        for image in images:
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image.data)
