import argparse
from pathlib import Path

def get_configs():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--ckpt_path',
                    default=None,
                    type=Path,
                    help='Checkpoint directory path')
    
    parser.add_argument('--save_path',
                    default=None,
                    type=Path,
                    help='Save directory path')
    
    parser.add_argument('--n_epochs',
                    default=5000,
                    type=int,
                    help='Number of total epochs to run')
    
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Batch Size')
    
    parser.add_argument('--train_shot',
                    default=16,
                    type=int,
                    help='Train Shot')
    
    parser.add_argument('--val_shot',
                default=4,
                type=int,
                help='Train Shot')
    
    parser.add_argument('--no_train',
                        action='store_true',
                        help='If true, training is not performed.')
    
    parser.add_argument('--no_val',
                        action='store_true',
                        help='If true, validation is not performed.')
    
    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')
    
    parser.add_argument('--checkpoint',
                        default=500,
                        type=int,
                        help='Trained model is saved at every this epochs.')

    cfg = parser.parse_args()

    return cfg