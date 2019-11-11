from pathlib import Path

import fire

import multifit.configurations
import multifit

class Experiment:
    def new(self):
        return {n: getattr(multifit.configurations,n) for n in multifit.configurations.__all__}

    def from_pretrained(self):
        return multifit.from_pretrained

if __name__ == '__main__':
    fire.Fire(Experiment())