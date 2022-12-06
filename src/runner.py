import json
from .core.hub import get_entries
# from .template_modules.losses import __mapping__ as loss_maps
# from .template_modules.metrics import __mapping__ as metric_maps
# from .template_modules.trainers import __mapping__ as trainer_maps
# from .template_modules.optimizers import __mapping__ as optimizer_map
from .models import __mapping__ as models_map
import os


class Runner:
    """
    create runner
    Args:
        config: config dict or path to config dict
        save_config_path: where to save config after training
        name: name of saved config
        verbose: print creation step

    Returns: ConfigRunner
    """

    def __init__(self, config, save_config_path=None, name=None, verbose=1):
        self.config = config
        self.save_config_path = save_config_path
        self.name = name or "train_config"
        self.global_config = config["global"]
        data_config = config["data"]
        model_config = config["model"]
        optimizer_config = config["optimizer"]
        trainer_config = config['trainer']
        print("creating train, valid loader") if verbose else None
        self.train, self.valid = self._get_data(data_config)
        # if verbose:
        #     print("train: ", len(self.train))
        #     print("valid: ", len(self.valid)) if self.valid is not None else None
        print("creating model") if verbose else None
        self.model = self._get_model(model_config)
        if verbose:
            save_model_txt_path = os.path.join(trainer_config['save_dir'], "model.txt")
            print("printing model to ", save_model_txt_path)
            with open(save_model_txt_path, "w") as handle:
                handle.write(str(self.model))
        self.optimizer = self._get_optimizer(optimizer_config)
        print("optimizer: ", self.optimizer) if verbose else None

        self.trainer = self._get_trainer(trainer_config)
        print("trainer ", self.trainer) if verbose else None

    def _get_data(self, data_config):
        if data_config['custom'] == True:
            return self._get_custom_module('data', data_config)
        else:
            raise NotImplementedError("No template config found for data")

    def _get_model(self, model_config):
        if model_config['custom'] == True:
            return self._get_custom_module('model', model_config)
        elif model_config['name'] in models_map:
            return self._get_default_module(models_map, model_config)
        else:
            raise NotImplementedError(
                "Invalid model config. Please specify either a custom model path or a valid model name")

    def _get_trainer(self, trainer_config):
        if trainer_config['custom'] == True:
            return self._get_custom_module(
                'trainer', trainer_config, args=[self.model, self.optimizer])
        else:
            raise NotImplementedError("No template config found for trainer")

    def _get_optimizer(self, optimizer_config):
        if optimizer_config['custom'] == True:
            return self._get_custom_module('trainer', optimizer_config, args=(self.model,))
        else:
            raise NotImplementedError("No template config found for trainer")

    def run(self):
        """
        run the training process based on config
        """
        self.trainer.train(self.train, self.valid)

        if self.save_config_path is not None:
            full_file = f"{self.save_config_path}/{self.name}"
            with open(full_file, "w") as handle:
                json.dump(self.config, handle)

    def __call__(self):
        self.run()

    @staticmethod
    def _get_kwargs(configs, excludes=("name",)):
        set_excludes = set(excludes)
        assert len(excludes) == len(set_excludes), 'Duplicated config found!'
        return {k: configs[k] for k in configs if k not in set_excludes}

    def _get_custom_module(self, module_name, config, excludes=("custom", "path", "method"), args=[]):
        entries = get_entries(config["path"])
        lentries = entries.list()
        assert config['method'] in lentries, 'Invalid method name in {} config'.format(module_name)
        kwargs = self._get_kwargs(config, excludes)
        return entries.load(config['method'], *args, **kwargs)

    def _get_default_module(self, module_map, config, excludes=("custom", "name"), args=[]):
        assert config['name'] in module_map, 'Invalid name in config'
        entries = module_map[config['name']]
        kwargs = self._get_kwargs(config, excludes)
        return entries(*args, **kwargs)

    @classmethod
    def from_dict(config, save_config_path=None, name=None, verbose=1):
        if not isinstance(config, dict):
            with open(config) as handle:
                config = json.load(handle)
        return Runner(config, save_config_path=save_config_path, name=name, verbose=verbose)
