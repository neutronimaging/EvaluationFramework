import numpy as np
from  ..core.dataset import Dataset
from ..core.image_generator import ImageGenerator

class ConstantLevelData(ImageGenerator):
    def __init__(self,  dims : list[int] = [100,100], name : str = "ConstantLevel"):
        super().__init__(name, dims)

    def _generate(self, batch_size : int, noise_level : float, params : dict = {}) -> list[ds.Dataset]:
        
        level = params.get("level", 0.5)
        seed  = params.get("seed", None)

        if seed is not None:
            np.random.seed(seed)
        
            
        return [Dataset(name=f"{self.name}_level={level}_noise={noise_level}",data=np.random.normal(level, noise_level, self._dims)) for _ in range(batch_size)]