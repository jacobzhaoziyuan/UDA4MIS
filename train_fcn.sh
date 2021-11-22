gpu=0
data=mr
# data2=ct
#model=fcn8s
model=drn26

datadir='D:/mmwhs_sifa/mmwhs_sifa/'
batch=8
epochs_num=50
lr=1e-3
momentum=0.99
num_cls=5
snapshot=5

#outdir=results/${data}-${data2}/${model}
outdir=results/${data}/${model}
mkdir -p results/${data}/${model}

python train_fcn.py ${outdir} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --epochs ${epochs_num} \
    --datadir ${datadir} \
    --snapshot $snapshot --dataset ${data}  #--dataset ${data2}
