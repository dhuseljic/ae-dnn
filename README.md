# AE-DNN
Code for reproducing some key results of our ICPR 2020 paper "Separation of Aleatoric and Epistemic
Uncertaintyin Deterministic Deep Neural Networks"

```bash
python experiment.py \
    --n_epochs $n_epochs \
    --n_reps $n_reps \
    --lmb $lmb \
    --ood_factor $ood_factor \
    --lr $lr \
    --weight_decay $weight_decay \
    --method_name $method_name \
    --dataset $dataset \
    --ood_ds $ood_ds \
    --gamma $gamma
```
