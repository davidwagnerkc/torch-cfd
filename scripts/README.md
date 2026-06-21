Low pass filter test:

```
  ../.venv/bin/python create_dataset.py --config-name 2dk_Re1k_test \
    downsample_method=spectral \
    dataset_name=2dk_re_sweep_spectral_test
```

```
../.venv/bin/python aggregate_dataset.py /scratch/dwcgt/2dk_re_sweep_spectral_test
```
