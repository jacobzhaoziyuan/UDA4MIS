
gpu=0

######################
# loss weight params #
######################
lr=1e-5
momentum=0.99
lambda_d=1
lambda_g=0.05

################
# train params #
################
max_iter=100000
crop=768
snapshot=100
batch=8

weight_share='weights_unshared'
discrim='discrim_feat'

########
# Data #
########
src='fake_ct'
tgt='ct'
datadir='D:/mmwhs_sifa/mmwhs_sifa'

# init with pre-trained cyclegta5 model
# model='drn26'
# baseiter=115000
model='fcn8s'
baseiter=100000


base_model="results/fake_ct/fcn8s/211110-1213/checkpoint.pth"
outdir="results/${src}_to_${tgt}/${model}"
mkdir -p results/${src}_to_${tgt}/${model}

# Run python script #
CUDA_VISIBLE_DEVICES=${gpu} python D:/cycada_2/cycada_2/train_fcn_adda.py \
    ${outdir} \
    --dataset ${src} --dataset ${tgt} --datadir ${datadir} \
    --lr ${lr} --momentum ${momentum} --gpu $gpu \
    --lambda_d ${lambda_d} --lambda_g ${lambda_g} \
    --weights_init ${base_model} --model ${model} \
    --"${weight_share}" --${discrim} --no_lsgan \
    --max_iter ${max_iter} --crop_size ${crop} --batch ${batch} \
    --snapshot $snapshot --num_cls 5
