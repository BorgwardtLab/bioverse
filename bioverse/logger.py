class Logger:

    def __init__(self, trainer):
        self.trainer = trainer

    @property
    def root(self):
        return self.trainer.root

    def log_loss(self, data, name=None):
        pass

    def log_dict(self, data):
        pass

    def log_tensor(self, data):
        pass

    def log_image(self, data):
        pass

    def log_text(self, data):
        pass
