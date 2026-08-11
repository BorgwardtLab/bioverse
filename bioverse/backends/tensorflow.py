import awkward as ak

from bioverse.backend import Backend


class TensorflowBackend(Backend):
    """TensorFlow/Keras training backend."""

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
        self._pending_checkpoint = None
        self._weights_restored = False

    @property
    def world_size(self):
        return 1

    @property
    def rank(self):
        return 0

    def _to_tensorflow(self, value, *, label=False):
        """Convert arrays to TensorFlow tensors, preserving integer labels."""
        import numpy as np

        tf = self.tf
        if label:
            if isinstance(value, tf.Tensor):
                array = value.numpy()
            else:
                array = ak.to_numpy(value)
            if np.issubdtype(array.dtype, np.integer):
                return tf.constant(array, dtype=tf.int32)
            return tf.constant(array, dtype=tf.float32)
        return ak.to_tensorflow(value)

    def _target_numpy(self, target):
        import numpy as np

        if hasattr(target, "numpy"):
            target = target.numpy()
        if np.issubdtype(target.dtype, np.floating) and np.all(target == np.floor(target)):
            return target.astype(np.int32)
        return target

    def put_on_device(self, data):
        if hasattr(data, "data1") and hasattr(data, "data2"):
            self.put_on_device(data.data1)
            self.put_on_device(data.data2)
            if getattr(data, "y", None) is not None:
                try:
                    data.y = self._to_tensorflow(data.y, label=True)
                except Exception:
                    pass
            return
        for key, value in vars(data).items():
            if not key.startswith("_"):
                try:
                    setattr(
                        data,
                        key,
                        self._to_tensorflow(value, label=(key == "y")),
                    )
                except Exception:
                    pass

    def _checkpoint_items(self, *, epoch=None, step=None, include_optimizer=False):
        tf = self.tf
        model = self.trainer.model
        items = {}
        if epoch is not None:
            items["epoch"] = epoch
        if step is not None:
            items["step"] = step
        if include_optimizer:
            items["optimizer"] = model.optimizer
        for idx, var in enumerate(model.checkpoint_variables):
            items[f"w{idx}"] = var
        return items

    def _maybe_restore_weights(self, Xy, data):
        if self._pending_checkpoint is None or self._weights_restored:
            return
        self.put_on_device(data)
        self.trainer.model.eval_step(Xy, data)
        ckpt = self.tf.train.Checkpoint(**self._checkpoint_items())
        ckpt.read(self._pending_checkpoint).expect_partial()
        self._weights_restored = True

    def train_step(self, Xy, data):
        self._maybe_restore_weights(Xy, data)
        self.put_on_device(data)
        with self.tf.GradientTape() as tape:
            loss, output = self.trainer.model.train_step(Xy, data)
        variables = getattr(self.trainer.model, "trainable_variables", [])
        grads = tape.gradient(loss, variables)
        self.trainer.model.optimizer.apply_gradients(zip(grads, variables))
        return loss.numpy(), output.numpy(), self._target_numpy(data.y)

    def eval_step(self, Xy, data):
        self._maybe_restore_weights(Xy, data)
        self.put_on_device(data)
        loss, output = self.trainer.model.eval_step(Xy, data)
        return loss.numpy(), output.numpy(), self._target_numpy(data.y)

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
            **self._checkpoint_items(
                epoch=epoch_var,
                step=step_var,
                include_optimizer=True,
            )
        )
        ckpt.write(path)

    def load_checkpoint(self, path):
        tf = self.tf
        self._pending_checkpoint = path
        self._weights_restored = False
        epoch_var = tf.Variable(0, dtype=tf.int64, trainable=False)
        step_var = tf.Variable(0, dtype=tf.int64, trainable=False)
        ckpt = tf.train.Checkpoint(epoch=epoch_var, step=step_var)
        ckpt.read(path).expect_partial()
        self.trainer.epoch = int(epoch_var.numpy())
        self.trainer.step = int(step_var.numpy())
