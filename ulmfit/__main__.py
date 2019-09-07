from pathlib import Path

import fire

import ulmfit.configurations
import ulmfit

class Experiment:
    def new(self):
        return {n: getattr(ulmfit.configurations,n) for n in ulmfit.configurations.__all__}

    def load(self, model_path):
        return ulmfit.ULMFiT().load_(Path(model_path))

    def download(self):
        raise NotImplementedError("implement model fetching")

if __name__ == '__main__':
    fire.Fire(Experiment())