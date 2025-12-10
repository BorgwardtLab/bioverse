import awkward as ak

from bioverse.backend import Backend


class TensorflowBackend(Backend):
    def __init__(
        self,
        trainer,
        accelerator="gpu",
        devices=1,
        strategy=None,
        precision=32,
        matmul_precision="medium",
        compile=False,
        clip_grad_norm=None,
        clip_grad_value=None,
        random_seed=None,
    ):
        import random

        import numpy as np
        import tensorflow as tf

        if random_seed is not None:
            random.seed(random_seed)
            tf.random.set_seed(random_seed)
            np.random.seed(random_seed)

        if devices > 1:
            raise ValueError("Tensorflow does not yet support multi-device training.")

        self.tf = tf
        self.trainer = trainer

    @property
    def world_size(self):
        return 1

    @property
    def rank(self):
        return 0

    def put_on_device(self, data):
        for key, value in vars(data).items():
            if not key.startswith("_"):
                try:
                    setattr(data, key, ak.to_tensorflow(value))
                except:
                    pass

    def train_step(self, Xy, data):
        self.put_on_device(data)
        with self.tf.GradientTape() as tape:
            loss, output = self.trainer.model.train_step(Xy, data)
        variables = getattr(self.trainer.model, "trainable_variables", [])
        grads = tape.gradient(loss, variables)
        self.trainer.model.optimizer.apply_gradients(zip(grads, variables))
        return loss.numpy(), output.numpy(), data.y.numpy()

    def eval_step(self, Xy, data):
        self.put_on_device(data)
        loss, output = self.trainer.model.eval_step(Xy, data)
        return loss.numpy(), output.numpy(), data.y.numpy()

    def pre_step(self, Xy, data):
        if not hasattr(self.trainer.model, "pre_step"):
            return
        self.put_on_device(data)
        self.trainer.model.pre_step(Xy, data)

    def save_checkpoint(self, path):
        tf = self.tf
        epoch_var = tf.Variable(self.trainer.epoch, dtype=tf.int64, trainable=False)
        step_var = tf.Variable(self.trainer.step, dtype=tf.int64, trainable=False)
        ckpt = tf.train.Checkpoint(
            model=self.trainer.model,
            optimizer=self.trainer.model.optimizer,
            epoch=epoch_var,
            step=step_var,
        )
        ckpt.write(path)

    def load_checkpoint(self, path):
        tf = self.tf
        epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
        step_var = tf.Variable(0, dtype=tf.int64, trainable=False)
        ckpt = tf.train.Checkpoint(
            model=self.trainer.model,
            optimizer=self.trainer.model.optimizer,
            epoch=epoch_var,
            step=step_var,
        )
        ckpt.read(path)
        self.trainer.epoch = int(ckpt.epoch.numpy())
        self.trainer.step = int(ckpt.step.numpy())
