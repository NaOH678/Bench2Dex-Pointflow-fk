"""Opt-in compilation with an explicit, permanent fallback for compiler failures.

Only known compiler errors trigger fallback. Model/input/CUDA errors still
propagate. Once compilation fails, this policy instance stays eager; there is
no per-frame retry storm, cache-limit increase, or silent mixed-mode timing.
"""


class CompiledForward:
    def __init__(self, eager, compiled, compiler_errors, on_fallback):
        self.eager = eager
        self.compiled = compiled
        self.compiler_errors = compiler_errors
        self.on_fallback = on_fallback
        self.failure = None

    def __call__(self, *args, **kwargs):
        if self.failure is not None:
            return self.eager(*args, **kwargs)
        try:
            return self.compiled(*args, **kwargs)
        except self.compiler_errors as exc:
            self.failure = dict(error_type=type(exc).__name__, error=str(exc))
            self.on_fallback(self.failure)
            return self.eager(*args, **kwargs)


def compile_forward(eager, *, dynamic=True, mode="default", on_fallback):
    import torch
    from torch._dynamo.exc import BackendCompilerFailed, FailOnRecompileLimitHit, Unsupported

    torch._inductor.config.emulate_precision_casts = True
    torch._inductor.config.force_same_precision = True
    # Variable point-cluster lengths make per-shape CUDA Graph recording costly.
    # Default mode retains Inductor compilation without CUDA Graph capture.
    compiled = torch.compile(eager, fullgraph=True, mode=mode, dynamic=dynamic)
    return CompiledForward(eager, compiled,
                           (BackendCompilerFailed, FailOnRecompileLimitHit, Unsupported),
                           on_fallback)
