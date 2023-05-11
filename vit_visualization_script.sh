./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type DENSE
mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/DENSE


# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type SRSTE --n-sparsity 2 --m-sparsity 4
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/SRSTE_NM_2_4
./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type SRSTE --n-sparsity 1 --m-sparsity 4 
mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/SRSTE_NM_1_4
# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type SRSTE --n-sparsity 1 --m-sparsity 64
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/SRSTE_NM_1_64
./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type SRSTE --n-sparsity 1 --m-sparsity 128
mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/SRSTE_NM_1_128

# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type STRUCTURED_NM --n-sparsity 2 --m-sparsity 4 
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_NM_2_4
# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type STRUCTURED_NM --n-sparsity 1 --m-sparsity 4 
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_NM_1_4
# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type STRUCTURED_NM --n-sparsity 1 --m-sparsity 64
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_NM_1_64
# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --sparsity-type STRUCTURED_NM --n-sparsity 1 --m-sparsity 128
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_NM_1_128


# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --decay-type EXP --n-sparsity 2 --m-sparsity 4 
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_EXP_NM_2_4
# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --decay-type LINEAR --n-sparsity 2 --m-sparsity 4 
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_LINEAR_NM_2_4


./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --decay-type EXP --n-sparsity 1 --m-sparsity 4 
mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_EXP_NM_1_4
./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --decay-type LINEAR --n-sparsity 1 --m-sparsity 4 
mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_LINEAR_NM_1_4

# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --decay-type EXP --n-sparsity 1 --m-sparsity 64
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_EXP_NM_1_64
./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --decay-type EXP --n-sparsity 1 --m-sparsity 128
mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_EXP_NM_1_128


# ./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --decay-type LINEAR --n-sparsity 1 --m-sparsity 64
# mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_LINEAR_NM_1_64
./distributed_train.sh 2 --config vit_small_trend_visulatization.yaml --decay-type LINEAR --n-sparsity 1 --m-sparsity 128
mv vit_model_stats_grad_sad_weights/2023* vit_model_stats_grad_sad_weights/STRUCT_LINEAR_NM_1_128
