# MNIST Config:
dataset='mnist_notmnist' 
lmb=0.024993225630577667 # 
ood_factor=8.744768627798972
gamma=4.412420740811563 # PriorNetGamma
weight_decay=1e-3
lr=1e-3

# SVHN Config:
# dataset='svhn_cifar10' 
# lmb=0.09091423290277234
# ood_factor=7.983537654015671
# gamma=1.3686354122657254 # PriorNetGamma
# weight_decay=1e-3
# lr=1e-3

# CIFAR% Config:
# dataset='cifar5_cifar5'
# lmb=0.0224033868117946
# ood_factor=9.109619573490875
# weight_decay=1e-3
# lr=1e-4

n_epochs=1
n_reps=1
ood_ds=0 # 0 or 1
for method_name in ordinary edl ensembles dropout xedl; do
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
done
