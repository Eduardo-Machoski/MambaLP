"""Wall-clock inference timing with explicit accelerator synchronization."""

from statistics import fmean, pstdev
from time import perf_counter


class InferenceBenchmark:
    def __init__(self, warmup_iterations=10, synchronize=None, clock=perf_counter):
        if isinstance(warmup_iterations, bool) or not isinstance(warmup_iterations, int) or warmup_iterations < 0:
            raise ValueError('test.warmup_iterations must be a nonnegative integer')
        self.warmup_iterations = warmup_iterations
        self.synchronize = synchronize or (lambda: None)
        self.clock = clock
        self.warmed_shapes = set()
        self.elapsed_ms = []
        self.image_count = 0

    def run(self, inference, shape):
        shape = tuple(shape)
        if shape not in self.warmed_shapes:
            for _ in range(self.warmup_iterations):
                inference()
            self.warmed_shapes.add(shape)
        # Pending transfers and warmup finish before the clock starts.
        self.synchronize()
        start = self.clock()
        result = inference()
        self.synchronize()
        self.elapsed_ms.append((self.clock() - start) * 1000.0)
        self.image_count += int(shape[0])
        return result

    def summary(self):
        return {
            'unit': 'ms_per_inference_call',
            'inference_count': len(self.elapsed_ms),
            'image_count': self.image_count,
            'mean_ms': fmean(self.elapsed_ms) if self.elapsed_ms else None,
            'std_ms': pstdev(self.elapsed_ms) if self.elapsed_ms else None,
            'std_definition': 'population (ddof=0)',
            'warmup_iterations_per_shape': self.warmup_iterations,
            'input_shapes': [list(shape) for shape in sorted(self.warmed_shapes)],
            'scope': 'model.test; excludes loading, device transfer, OCR, metrics and saving',
        }
