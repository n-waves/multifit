from pathlib import Path

import fire
from dataclasses import asdict

import ulmfit.configurations
import ulmfit

class Experiment:
    def new(self):
        return {n: getattr(ulmfit.configurations,n) for n in ulmfit.configurations.__all__}

    def load(self, model_path):
        return ulmfit.ULMFiT().load_(Path(model_path))

    def download(self):
        raise NotImplementedError("implement model fetching")

    def evaluate(self, glob):
        return ExperimentList(self.load(model_path) for base_path in sorted(Path.cwd.glob(glob))
                   for model_path in base_path.glob("**/*.m"))

if __name__ == '__main__':
    fire.Fire(Experiment())