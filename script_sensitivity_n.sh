flag=sensitivity

data=self_data_pcl
model=pclad

for n_trans in 2 3 7;do

    echo $n_trans
    dataset=01thyroid

    python -u main.py \
    --input_dir $data \
    --dataset $dataset --act LeakyReLU --n_trans $n_trans \
    --flag $flag \
    > log/${model}-${flag}.n_trans${n_trans}.1.log 2>&1 &

    wait
done

for n_trans in 2 3 4 7 15;do

    echo $n_trans
    dataset=02arrhythmia,03bank,04celeba,05secom,06UNSW_NB15_traintest_Analysis,07UNSW_NB15_traintest_Backdoor,08UNSW_NB15_traintest_DoS,09UNSW_NB15_traintest_Exploits,10UNSW_NB15_traintest_Fuzzers,11UNSW_NB15_traintest_Generic,12UNSW_NB15_traintest_Reconnaissance,13Tuesday_drop_data_label,14Wednesday_drop_data_label,15Thursday_drop_data_label,16Friday_drop_data_label

    python -u main.py \
    --input_dir $data \
    --dataset $dataset --act LeakyReLU --n_trans $n_trans \
    --flag $flag \
    > log/${model}-${flag}.n_trans${n_trans}.2.log 2>&1 &

    wait
done