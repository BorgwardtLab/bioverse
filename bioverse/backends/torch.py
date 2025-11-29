import awkward as ak

from bioverse.backend import Backend


class TorchBackend(Backend):
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
    ):
        import torch
        from lightning.fabric import Fabric

        torch.set_float32_matmul_precision(matmul_precision)
        self.torch = torch
        self.clip_grad_norm = clip_grad_norm

        self.trainer = trainer
        self.fabric = Fabric(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
        )
        if devices > 1:
            self.fabric.launch()
        self.trainer.model, self.trainer.optimizer = self.fabric.setup(
            torch.compile(self.trainer.model) if compile else self.trainer.model,
            self.trainer.model.optimizer,
        )
        if clip_grad_value is not None:
            for p in self.trainer.model.parameters():
                p.register_hook(
                    lambda grad: torch.clamp(grad, -clip_grad_value, clip_grad_value)
                )

        # Mark custom step methods as forward methods for Fabric-wrapped models
        if hasattr(self.trainer.model, "mark_forward_method"):
            for method_name in ("train_step", "eval_step", "pre_step"):
                if hasattr(self.trainer.model, method_name):
                    self.trainer.model.mark_forward_method(method_name)

    @property
    def world_size(self):
        return self.fabric.world_size

    @property
    def rank(self):
        return self.fabric.global_rank

    def put_on_device(self, data):
        for key, value in vars(data).items():
            if not key.startswith("_"):
                try:
                    setattr(data, key, ak.to_torch(value).to(self.fabric.device))
                except:
                    pass

    def train_step(self, Xy, data):
        self.put_on_device(data)
        self.trainer.model.train()
        self.trainer.model.optimizer.zero_grad()
        loss, output = self.trainer.model.train_step(Xy, data)
        self.fabric.backward(loss)
        if self.clip_grad_norm is not None:
            self.torch.nn.utils.clip_grad_norm_(
                self.trainer.model.parameters(), self.clip_grad_norm
            )
        self.trainer.model.optimizer.step()
        if hasattr(self.trainer.model, "scheduler"):
            self.trainer.model.scheduler.step()
        return loss.item(), output.detach().cpu(), data.y.detach().cpu()

    def eval_step(self, Xy, data):
        self.put_on_device(data)
        self.trainer.model.eval()
        with self.torch.no_grad():
            loss, output = self.trainer.model.eval_step(Xy, data)
        return loss.item(), output.detach().cpu(), data.y.detach().cpu()

    def pre_step(self, Xy, data):
        if not hasattr(self.trainer.model, "pre_step"):
            return
        self.put_on_device(data)
        self.trainer.model.eval()
        with self.torch.no_grad():
            self.trainer.model.pre_step(Xy, data)

    def save_checkpoint(self, path):
        state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "epoch": self.trainer.epoch,
            "step": self.trainer.step,
        }
        self.fabric.save(path.with_suffix(".pt"), state)

    def load_checkpoint(self, path):
        state = self.fabric.load(path.with_suffix(".pt"))
        self.trainer.model.load_state_dict(state["model"])
        self.trainer.optimizer.load_state_dict(state["optimizer"])
        self.trainer.epoch = state["epoch"]
        self.trainer.step = state["step"]
