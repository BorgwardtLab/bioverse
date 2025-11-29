from abc import abstractmethod


class Collater:

    def __call__(self, *args, **kwargs):
        return self.collate(*args, **kwargs)

    @abstractmethod
    def collate(self, *args, **kwargs):
        raise NotImplementedError
