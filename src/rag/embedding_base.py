import numpy as np
from typing import List
from abc import ABC, abstractmethod

class EmbeddingBase(ABC):
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        pass
