import numpy as np

from abc import ABC, abstractmethod


class AbstractPolicy(ABC):
    """
    Common interface for any Agent Policy.

    Point to notice:
        In common, Policy could be non-learnable. For this reason, function learn() is absent here.    

    Author:
        Darr Mirr (Vladimir S. Polukeev)
    """

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> int:
        pass
