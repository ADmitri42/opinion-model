from typing import Dict
import numpy as np
from .opinion import Opinion


def _opinions_to_array(opinions: Dict[str, Opinion]) -> np.ndarray:
    return np.fromiter(
        (t[-1] for t in sorted(opinions.items(), key=lambda item: (int(item[0]), item[1]))),
        int
    )
