from pathlib import Path

from bioverse import backends, collaters, loggers
from bioverse.utilities import config, info, save


def finite_score(score):
    """Return ``float(score)`` or ``None`` if the value is missing or NaN."""
    if score is None:
        return None
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    if score != score:
        return None
    return score


def score_improved(score, best, *, higher=True, min_delta=0.0):
    """Whether ``score`` improves on ``best`` for the monitored metric."""
    score = finite_score(score)
    if score is None:
        return False
    if best is None:
        return True
    best = float(best)
    if higher:
        return score > best + min_delta
    return score < best - min_delta


class Trainer:
    """High-level training and evaluation loop for Bioverse benchmarks.

    The trainer wires a user model to a :class:`~bioverse.benchmark.Benchmark`
    through a :class:`~bioverse.backend.Backend`, :class:`~bioverse.collater.Collater`,
    and :class:`~bioverse.logger.Logger`. It handles checkpointing, validation
    scheduling, and distributed training configuration.

    Instantiate from a YAML experiment config or directly in Python, then call
    :meth:`run` with ``"train"``, ``"val"``, or ``"test"``.

    Examples
    --------
    From the CLI (recommended):

    .. code-block:: bash

       bioverse train experiment.yaml

    In Python:

    .. code-block:: python

       from bioverse import Trainer
       from bioverse.factory import BenchmarkFactory

       benchmark = BenchmarkFactory("B_AFCATH")
       trainer = Trainer(model, benchmark, epochs=100, batch_size=32)
       trainer.run("train")
    """

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
        restore_strict=True,
        restore_optimizer=True,
        save_best=True,
        checkpoint_metric=None,
        patience=None,
        min_delta=0.0,
        log_every=100,
        epochs=1,
        batch_size=32,
        batch_on="molecules",
        shuffle=True,
        drop_last=False,
        random_seed=42,
        progress=True,
        scratch=False,
        train_partition_name="train",
        val_partition_name="val",
        test_partition_name="test",
        pre_partition_name="pre",
        limit_train_batches=None,
    ):
        config.workers = workers
        config.seed = random_seed

        self.root = Path(root)
        self.model_name = model_name
        self.epochs = epochs
        self.checkpoint_every = checkpoint_every
        self.checkpoint_name = checkpoint_name
        self.restore_from = restore_from
        self.restore_strict = bool(restore_strict)
        self.restore_optimizer = bool(restore_optimizer)
        self.save_best = save_best
        self.checkpoint_metric = checkpoint_metric
        self.patience = None if patience is None or patience <= 0 else int(patience)
        self.min_delta = float(min_delta)
        self._best_score = None
        self._epochs_without_improvement = 0
        self.log_every = log_every
        self.batch_size = batch_size
        self.batch_on = batch_on
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.loader_attr = loader_attr
        self.random_seed = random_seed
        self.progress = progress
        self.scratch = scratch
        self.train_partition_name = train_partition_name
        self.val_partition_name = val_partition_name
        self.test_partition_name = test_partition_name
        self.pre_partition_name = pre_partition_name
        self.initial_validation = initial_validation
        self.limit_train_batches = limit_train_batches

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

    def _maybe_restore(self):
        path = self.root / self.restore_from
        if path.with_suffix(".pt").exists():
            self.backend.load_checkpoint(path)

    def run_train(self):
        self._maybe_restore()
        if not self.val_partition_name is None and self.initial_validation:
            self.run_eval("val")
        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            loader = self.benchmark.loader(
                partition=self.train_partition_name,
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
            for batch_idx, (Xy, data) in enumerate(loader):
                self.step += 1
                loss, output, target = self.backend.train_step(Xy, data)
                if self.step % self.log_every == 0:
                    self.logger.log_loss(loss, mode="train")
                if (
                    self.limit_train_batches is not None
                    and batch_idx + 1 >= self.limit_train_batches
                ):
                    break
            if not self.val_partition_name is None:
                result = self.run_eval("val")
                improved = self._update_best(result)
                if self.save_best and improved:
                    self.backend.save_checkpoint(
                        self.root / (self.checkpoint_name.format(epoch=epoch))
                    )
                if self.patience is not None:
                    if improved:
                        self._epochs_without_improvement = 0
                    else:
                        self._epochs_without_improvement += 1
                        if self._epochs_without_improvement >= self.patience:
                            info(
                                f"Early stopping at epoch {epoch} "
                                f"({self.patience} epochs without improvement)."
                            )
                            break
            if (
                (not self.save_best or self.val_partition_name is None)
                and isinstance(self.checkpoint_every, int)
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
            partition={
                "val": self.val_partition_name,
                "test": self.test_partition_name,
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
        return result

    def _monitor_score(self, result):
        scores = result.to_dict().get(self.model_name) or {}
        if not scores:
            return None, None, True
        metric = self.checkpoint_metric
        if metric is None or metric not in scores:
            metric = next(iter(scores))
        direction = result.better.get(metric, "higher")
        return metric, finite_score(scores.get(metric)), direction == "higher"

    def _update_best(self, result):
        _, score, higher = self._monitor_score(result)
        improved = score_improved(
            score, self._best_score, higher=higher, min_delta=self.min_delta
        )
        if improved:
            self._best_score = score
        return improved

    def run_pre(self):
        loader = self.benchmark.loader(
            partition=self.pre_partition_name,
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
