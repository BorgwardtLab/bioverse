from pathlib import Path

from bioverse import backends, collaters, loggers
from bioverse.utilities import config, save


class Trainer:
    def __init__(
        self,
        model,
        benchmark,
        backend="TorchBackend",
        collater="LongCollater",
        logger="DiskLogger",
        root="results",
        model_name="Ours",
        validate_every=1.0,
        initial_validation=False,
        accelerator="cpu",
        devices=1,
        workers=1,
        loader_attr=[],
        precision=32,
        matmul_precision="medium",
        strategy="auto",
        compile=False,
        accumulate_gradients=1,
        clip_grad_norm=None,
        clip_grad_value=None,
        checkpoint_every=1.0,
        checkpoint_name="checkpoint",
        restore_from="checkpoint",
        log_every=100,
        epochs=1,
        batch_size=32,
        batch_on="molecules",
        shuffle=True,
        drop_last=False,
        random_seed=42,
        progress=True,
        scratch=False,
        train_split_name="train",
        val_split_name="val",
        test_split_name="test",
        pre_split_name="pre",
    ):
        config.workers = workers
        config.seed = random_seed

        self.root = Path(root)
        self.model_name = model_name
        self.epochs = epochs
        self.checkpoint_every = checkpoint_every
        self.checkpoint_name = checkpoint_name
        self.restore_from = restore_from
        self.log_every = log_every
        self.batch_size = batch_size
        self.batch_on = batch_on
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.loader_attr = loader_attr
        self.random_seed = random_seed
        self.progress = progress
        self.scratch = scratch
        self.train_split_name = train_split_name
        self.val_split_name = val_split_name
        self.test_split_name = test_split_name
        self.pre_split_name = pre_split_name
        self.initial_validation = initial_validation

        self.benchmark = benchmark
        self.model = model
        self.backend = getattr(backends, backend)(
            self,
            accelerator,
            devices,
            strategy,
            precision,
            matmul_precision,
            compile,
            clip_grad_norm,
            clip_grad_value,
            random_seed,
        )
        self.collater = getattr(collaters, collater)()
        if isinstance(logger, dict):
            logger, logger_kwargs = next(iter(logger.items()))
        else:
            logger_kwargs = {}
        self.logger = getattr(loggers, logger or "NoLogger")(self, **logger_kwargs)
        self.epoch = 0
        self.step = 0

    def run(self, command):
        if command == "train":
            self.run_train()
        else:
            self.run_eval(command)

    def run_train(self):
        if not self.val_split_name is None and self.initial_validation:
            self.run_eval("val")
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            loader = self.benchmark.loader(
                split=self.train_split_name,
                collater=self.collater,
                batch_size=self.batch_size,
                batch_on=self.batch_on,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
                random_seed=self.random_seed + epoch,
                world_size=self.backend.world_size,
                rank=self.backend.rank,
                progress=self.progress,
                scratch=self.scratch,
                attr=self.loader_attr,
            )
            for Xy, data in loader:
                self.step += 1
                loss, output, target = self.backend.train_step(Xy, data)
                if self.step % self.log_every == 0:
                    self.logger.log_loss(loss, mode="train")
            if not self.val_split_name is None:
                self.run_eval("val")
            if (
                isinstance(self.checkpoint_every, int)
                and epoch % self.checkpoint_every == 0
            ):
                self.backend.save_checkpoint(
                    self.root / (self.checkpoint_name.format(epoch=epoch))
                )

    def run_eval(self, command):
        self.run_pre()
        if command == "test":
            self.backend.load_checkpoint(self.root / self.restore_from)
        loader = self.benchmark.loader(
            split={
                "val": self.val_split_name,
                "test": self.test_split_name,
            }[command],
            collater=self.collater,
            batch_size=self.batch_size,
            batch_on=self.batch_on,
            shuffle=False,
            drop_last=False,
            random_seed=self.random_seed,
            world_size=self.backend.world_size,
            rank=self.backend.rank,
            progress=self.progress,
            scratch=self.scratch,
            attr=self.loader_attr,
        )
        for Xy, data in loader:
            loss, output, target = self.backend.eval_step(Xy, data)
            output = data.uncollate(output)
            target = data.uncollate(target)
            self.benchmark.update(target, output)
        # todo: logging & eval with specified intervals
        result = self.benchmark.result(model_name=self.model_name)
        result.to_console()
        self.logger.log_dict(result.to_dict()[self.model_name], mode=command)
        save(result.to_dict()[self.model_name], self.root / f"{command}_results.yaml")

    def run_pre(self):
        loader = self.benchmark.loader(
            split=self.pre_split_name,
            collater=self.collater,
            batch_size=self.batch_size,
            batch_on=self.batch_on,
            shuffle=False,
            drop_last=False,
            random_seed=self.random_seed,
            world_size=self.backend.world_size,
            rank=self.backend.rank,
            progress=self.progress,
            scratch=self.scratch,
            attr=self.loader_attr,
        )
        if not loader is None:
            for Xy, data in loader:
                self.backend.pre_step(Xy, data)
