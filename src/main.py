import hydra
from omegaconf import DictConfig

from src.training_module import sa_trainer_csl

@hydra.main(version_base=None, config_path="../conf", config_name="conf")
def main(cfg: DictConfig):

    sa_trainer_csl.run_sequence_classification_trainig_eval_test(cfg)
        

if __name__ == "__main__":
    main()
