flag=robustness

data=self_data_pcl
model=pclad


for cont in 0.005 0.01 0.015 0.02 0.025 0.03;do

    echo $cont
    dataset=01thyroid
    n_trans=4
    python -u main.py \
    --input_dir $data --contamination $cont \
    --dataset $dataset --act LeakyReLU --n_trans $n_trans \
    --flag $flag \
    > log/${model}-${flag}.cont${cont}.1.log 2>&1 &

    dataset=02arrhythmia,03bank,04celeba,05secom,06UNSW_NB15_traintest_Analysis,07UNSW_NB15_traintest_Backdoor,08UNSW_NB15_traintest_DoS,09UNSW_NB15_traintest_Exploits,10UNSW_NB15_traintest_Fuzzers,11UNSW_NB15_traintest_Generic,12UNSW_NB15_traintest_Reconnaissance,13Tuesday_drop_data_label,14Wednesday_drop_data_label,15Thursday_drop_data_label,16Friday_drop_data_label
    n_trans=11
    python -u main.py \
    --input_dir $data --contamination $cont \
    --dataset $dataset --act LeakyReLU --n_trans $n_trans \
    --flag $flag \
    > log/${model}-${flag}.cont${cont}.2.log 2>&1 &

    wait

done

