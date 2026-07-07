"""
Multi-dataset wrapper to concatenate several datasets defined in YAML configs.
"""

from torch.utils.data import ConcatDataset as TorchConcatDataset

from ...core import register, create, GLOBAL_CONFIG


@register()
class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets_list):
        # Build each sub-dataset from the registry
        datasets, dataset_names = self._build_datasets(datasets_list)

        # Cache lengths for later splitting during evaluation
        self.sub_dataset_len = [len(ds) for ds in datasets]
        self.dataset_names = dataset_names

        super().__init__(datasets)

    def _build_datasets(self, datasets_list):
        datasets, dataset_names = [], []
        for name, cfg in datasets_list.items():
            # Merge specific config with defaults from registry
            _cfg = GLOBAL_CONFIG[cfg['type']].copy()
            _cfg.update(cfg)
            # `create` consumes kwargs from the global cfg entry (without a `type` key)
            _cfg.pop('type', None)

            # Use a local copy of the registry so per-dataset args (paths, transforms, etc.)
            # are passed through without polluting global defaults.
            local_registry = GLOBAL_CONFIG.copy()
            local_registry[cfg['type']] = _cfg
            print(f'building {name} dataset')
            datasets.append(create(cfg['type'], local_registry))
            dataset_names.append(name)
        return datasets, dataset_names

    def set_epoch(self, epoch):
        # Propagate epoch to sub-datasets if they support it
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
