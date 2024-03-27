from src.argument import parse_args
from src.utils import set_random_seeds
from models import SGCL_Trainer
import torch


def main():
    args = parse_args()
    set_random_seeds(0)
    torch.set_num_threads(2)
    embedder = SGCL_Trainer(args)
    embedder.train()


if __name__ == "__main__":
    main()


