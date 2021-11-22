
gpu=2

######################
# loss weight params #
######################
lr=1e-5
momentum=0.99
lambda_d=1
lambda_g=0.1

################
# train params #
################
max_iter=1000
crop=768
snapshot=5
batch=1

weight_share='weights_shared'
discrim='discrim_score'

########
# Data #
########
src='mr'
tgt='ct'
datadir='/home/ziyuan/UDA/data/mmwhs_sifa/'

# init with pre-trained cyclegta5 model
# model='drn26'
# baseiter=115000
model='fcn8s'
baseiter=100000


base_model="results/mr/fcn8s/211028-0313/iter35.pth"
outdir="results/${src}_to_${tgt}/${model}"
mkdir -p results/${src}_to_${tgt}/${model}

# Run python script #
CUDA_VISIBLE_DEVICES=${gpu} python scripts/train_fcn_adda.py \
    ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} --gpu $gpu \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --weights_init ${base_model} --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot $snapshot --num_cls 5
