import numpy as np
from  ..core.dataset import Dataset
from ..core.image_generator import ImageGenerator
import ..io.readers as rd

class ImageLoader(ImageGenerator) :
    def __init__(self,  dims : list[int] = [100,100], name : str = "ImageLoader"):
        super().__init__(name, dims)

    def _generate(self, batch_size : int, noise_level : float, params : dict = {}) -> list[Dataset]:
        
        image_paths = params.get("image_paths", [])
        if not image_paths:
            raise ValueError("image_paths must be provided in params for ImageLoader")

        datasets = []
        for path in image_paths:
            image_data = self._load_image(path)
            noisy_image_data = image_data + np.random.normal(0, noise_level, self._dims)
            datasets.append(Dataset(name=f"{self.name}_{path}_noise={noise_level}", data=noisy_image_data, ground_truth=image_data))
        
        return datasets

    def _load_image(self, path):
        # Placeholder for actual image loading logic
        # In a real implementation, you would use an image processing library like PIL or OpenCV to load the image
        return rd.read_image(path)
