class EarlyStopping:
    """
    Stops training when a monitored metric stops improving.

    Args:
        patience:      number of consecutive val checks with no improvement before stopping
        min_delta:     minimum change to count as improvement
        warmup_steps:  ignore early stopping until this many steps have passed
                       (useful for VAEs during KL annealing phase)
        mode:          "max" for metrics like Dice/SSIM, "min" for loss
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 1e-4,
        warmup_steps: int = 0,
        mode: str = "max",
    ) -> None:
        self.patience      = patience
        self.min_delta     = min_delta
        self.warmup_steps  = warmup_steps
        self.mode          = mode
        self.best: float | None = None
        self.counter       = 0

    def step(self, metric: float, current_step: int) -> bool:
        """
        Call after each val check. Returns True if training should stop.
        """
        if current_step < self.warmup_steps:
            return False

        if self.best is None:
            self.best = metric
            return False

        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    @property
    def status(self) -> str:
        best_str = f"{self.best:.4f}" if self.best is not None else "n/a"
        return f"patience {self.counter}/{self.patience}  best={best_str}"
