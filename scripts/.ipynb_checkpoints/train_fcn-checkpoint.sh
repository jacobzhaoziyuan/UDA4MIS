gpu=2
data=mr
data2=ct
model=fcn8s


crop=768
datadir='/home/ziyuan/UDA/data/mmwhs_sifa/'
batch=2
iterations=5000
lr=1e-3
momentum=0.99
num_cls=5
snapshot=5

#outdir=results/${data}-${data2}/${model}
outdir=results/${data}/${model}
mkdir -p results/${data}/${model}

python scripts/train_fcn.py ${outdir} --model ${model} \
    --num_cls ${num_cls} --gpu ${gpu} \
    --lr ${lr} -b ${batch} -m ${momentum} \
    --crop_size ${crop} --iterations ${iterations} \
    --datadir ${datadir} \
    --snapshot $snapshot --dataset ${data}  #--dataset ${data2}
