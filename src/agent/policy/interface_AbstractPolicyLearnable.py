from agent.policy.interface_AbstractPolicy import AbstractPolicy
from abc import abstractmethod


class AbstractPolicyLearnable(AbstractPolicy):
    """
    Interface for Agent learnable Policy.  

    Author:
        Darr Mirr (Vladimir S. Polukeev)
    """

    @abstractmethod
    def learn(self, train_samples: list[tuple]) -> float:
        pass

    @abstractmethod
    def export_to_file(self,
                       target_dir: str,
                       file_prefix: str = '',
                       file_suffix: str = ''):
        pass

    @abstractmethod
    def import_from_file(self, file_path: str):
        pass
