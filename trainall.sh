# Malware
python train.py --config config/malware.yml --model_name DeepFM --optimize_hyperparameters
python train.py --config config/malware.yml --model_name VIME --optimize_hyperparameters
python train.py --config config/malware.yml --model_name TabTransformer --optimize_hyperparameters
python train.py --config config/malware.yml --model_name TORCHRLN --optimize_hyperparameters
python train.py --config config/malware.yml --model_name ModelTree --optimize_hyperparameters

# URL
python train.py --config config/url.yml --model_name DeepFM --optimize_hyperparameters
python train.py --config config/url.yml --model_name VIME --optimize_hyperparameters
python train.py --config config/url.yml --model_name TabTransformer --optimize_hyperparameters

# WIDS
python train.py --config config/wids.yml --model_name DeepFM --optimize_hyperparameters
python train.py --config config/wids.yml --model_name VIME --optimize_hyperparameters
python train.py --config config/wids.yml --model_name TabTransformer --optimize_hyperparameters

# LCLD
python train.py --config config/lcld_v2_time.yml --model_name DeepFM --optimize_hyperparameters
python train.py --config config/lcld_v2_time.yml --model_name VIME --optimize_hyperparameters
python train.py --config config/lcld_v2_time.yml --model_name TabTransformer --optimize_hyperparameters
python train.py --config config/lcld_v2_time.yml --model_name TORCHRLN --optimize_hyperparameters


# CTU
python train.py --config config/ctu_13_neris.yml --model_name DeepFM --optimize_hyperparameters
python train.py --config config/ctu_13_neris.yml --model_name VIME --optimize_hyperparameters
python train.py --config config/ctu_13_neris.yml --model_name TORCHRLN --optimize_hyperparameters


