
def load_dataset(cfg, mode=''):

    if 'FiveK' in cfg.dataset_name:
        resol = cfg.testing['resol']
        if 'Retouch' in cfg.task_name:
            from dataloaders.Retouching.FiveK_dataset import FiveK_train_dataset, FiveK_test_dataset

            dataset = FiveK_train_dataset(cfg, expert=f'{cfg.expert}') if mode == 'train' \
                else FiveK_test_dataset(cfg, expert=f'{cfg.expert}', resol=f'{resol}')
        else:
            from dataloaders.ToneMap.FiveK_dataset import FiveK_train_dataset, FiveK_test_dataset

            dataset = FiveK_train_dataset(cfg, expert=f'{cfg.expert}') if mode == 'train' \
                else FiveK_test_dataset(cfg, expert=f'{cfg.expert}', resol=f'{resol}')

    elif 'PPR10K' in cfg.dataset_name:
        from dataloaders.Retouching.PPR10K_dataset import PPR10K_train_dataset, PPR10K_test_dataset

        dataset = PPR10K_train_dataset(cfg, expert=f'{cfg.expert}') if mode == 'train'\
            else PPR10K_test_dataset(cfg, expert=f'{cfg.expert}')

    elif 'UIEB' in cfg.dataset_name:
        from dataloaders.Underwater.UIEB_dataset import UIEB_train_dataset, UIEB_test_dataset
        dataset = UIEB_train_dataset(cfg) if mode == 'train' else UIEB_test_dataset(cfg)

    elif 'EUVP' in cfg.dataset_name:
        from dataloaders.Underwater.EUVP_dataset import EUVP_train_dataset, EUVP_test_dataset
        dataset = EUVP_train_dataset(cfg) if mode == 'train' else EUVP_test_dataset(cfg)

    return dataset
