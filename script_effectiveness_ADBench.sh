flag=effectiveness.ADBench

data=self_ADBench
model=pclad
dataset=FULL
n_trans=11
e=20

python -u main.py \
--input_dir $data \
--dataset $dataset \
--act LeakyReLU --n_trans $n_trans --epochs $e \
--flag $flag \
> log/${model}-${flag}.1.log 2>&1 &