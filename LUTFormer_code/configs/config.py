import os
import yaml


def write_log(log_file, out_str):
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)


class Config:
    def __init__(self):
        self.root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.gpu = '0'

        # run_mode : select one in ['train', 'test', 'runtime', 'test_paper']
        self.run_mode = 'test_paper'
        self.load = True            # Load pre-trained model
        self.viz = False

        # model_name  : dataset_name
        # 'Retouching': ['FiveK', 'PPR10K']
        # 'ToneMap':    ['FiveK']
        # 'Underwater': ['UIEB', 'EUVP']
        self.model_name = 'Retouching'
        self.dataset_name = 'FiveK'
        self.expert = 'c'

        if self.dataset_name == 'FiveK':
            self.expert = 'c'
        if self.model_name == 'Underwater':
            self.expert = ''

        self.validate_config()
        self.override_config_with_yaml(f'configs/{self.model_name}_{self.dataset_name}.yaml')
        self.settings_for_path()

    def settings_for_path(self):
        # Set your dataset root
        self.dataset_root = os.path.abspath(os.path.join('/home/jwko/Datasets'))

        self.code_name = os.getcwd().split('/')[-1]
        self.output_name = self.code_name.replace('_code', '_output')
        self.proj_dir = os.path.join(self.root, f'{self.code_name}')
        self.output_dir = os.path.join(self.root, f'{self.output_name}')

        self.exp_name = f'{self.model_name}_{self.dataset_name}_expert{self.expert.capitalize()}'
        self.viz_dir = os.path.join(self.output_dir, f'display/{self.exp_name}')
        self.save_dir = os.path.join(self.output_dir, f'weights/{self.exp_name}')
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self.checkpoint = {}
        if self.load:
            ckpt_name = 'ckpt/checkpoint_best.pth'
            self.checkpoint['model'] = os.path.join(self.save_dir, ckpt_name)

        if self.run_mode == 'test_paper':
            ckpt_dir = os.path.join(self.code_name, f'pretrained')
            ckpt_name = f'{self.exp_name}.pth'
            self.checkpoint['model'] = os.path.join(ckpt_dir, ckpt_name)

    def override_config_with_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            override_cfg = yaml.safe_load(f)
        self._recursive_update(self.__dict__, override_cfg)

    def _recursive_update(self, base, overrides):
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                self._recursive_update(base[k], v)
            else:
                base[k] = v

    def log_configs(self, log_file='log.txt'):
        if os.path.exists(f'{self.save_dir}/{log_file}'):
            log_file = open(f'{self.save_dir}/{log_file}', 'a')
        else:
            log_file = open(f'{self.save_dir}/{log_file}', 'w')

        write_log(log_file, '------------ Options -------------')
        for k in vars(self):
            write_log(log_file, f'{str(k)}: {str(vars(self)[k])}')
        write_log(log_file, '-------------- End ----------------')

        log_file.close()
        return

    def validate_config(self):
        valid_models = {'Retouching': ['FiveK', 'PPR10K'],
                        'ToneMap': ['FiveK'],
                        'Underwater': ['UIEB', 'EUVP']}

        valid_experts = {'FiveK': ['c'],
                         'PPR10K': ['a', 'b', 'c'],
                         'UIEB': [''],
                         'EUVP': ['']}

        if self.model_name not in valid_models:
            raise ValueError(f'Invalid model_name: {self.model_name}')

        if self.dataset_name not in valid_models[self.model_name]:
            raise ValueError(f'Invalid dataset_name: {self.dataset_name}')

        if self.expert not in valid_experts[self.dataset_name]:
            raise ValueError(f'Invalid expert: {self.expert}')
