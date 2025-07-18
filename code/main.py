import os

from configs.config import Config
from models.networks.get_model import get_model
from engines.loss import get_loss_function
from engines.optimizer import get_optimizer, get_scheduler
from engines.get_train_process import do_train_process
from engines.get_test_process import do_test_process, do_check_runtime

from utils.util import copy_code


if __name__ == '__main__':
    args = Config()
    args.log_configs()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = get_model(args)
    criterion = get_loss_function(args)

    if 'test' in args.run_mode:   # For test
        do_test_process(args, model)
    elif 'train' in args.run_mode:
        copy_code(args)
        optimizer = get_optimizer(args, model)
        lr_scheduler = get_scheduler(args, optimizer)
        do_train_process(args, model, criterion, optimizer, lr_scheduler)
    elif 'runtime' in args.run_mode:
        do_check_runtime(args, model)
    elif 'test_paper' in args.run_mode:   # For test
        do_test_process(args, model)