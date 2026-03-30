import numpy as np
from  ..core.dataset import Dataset
from ..core.image_generator import ImageGenerator
from ..io import readers as rd

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
            tmpname=path.split('/')[-1].split('.')[0]
            if self._dims is None :
                datasets.append(
                    Dataset(
                        name=f"{self.name}_{tmpname}",
                        data=image_data
                    )
                )
            else :
                for k, patch in enumerate(self._make_patches(image_data)):            
                    datasets.append(
                        Dataset(
                            name=f"{self.name}_{tmpname}_patch_{k:04d}",
                            data=patch
                    )
                )
        
        return datasets

    def _load_image(self, path):
        # Placeholder for actual image loading logic
        # In a real implementation, you would use an image processing library like PIL or OpenCV to load the image
        return rd.read_image(path)
    
    def _make_patches(self, image):
        r = image.shape[0] // self._dims[0]
        c = image.shape[1] // self._dims[1]

        if r == 0 or c == 0:
            raise ValueError("Image dimensions are smaller than patch size")    
         
        for i in range(r):
            for j in range(c):
                yield image[i*self._dims[0]:(i+1)*self._dims[0], j*self._dims[1]:(j+1)*self._dims[1]]
