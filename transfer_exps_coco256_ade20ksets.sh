## to compute confusion matrix
# for i in {1..9}
# do
# 	python test.py  --name coco/100%TrainData --dataset_mode ade20k --gpu_ids 0 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 10 --size 256 --disc_out --subset_nb $i --use_subset 
# done


############################################ trainAll #########################################
python trainer_submitit_transfer.py --name ade20kSets_transfer/coco256/classical_finet/trainAll  --dataset_mode ade20k --gpu_ids 0 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 256 --freq_fid 100 --use_subset --trainG --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer

python trainer_submitit_transfer.py --name ade20kSets_transfer/coco256/classical_finet/trainAll_cosineLrSched  --dataset_mode ade20k --gpu_ids 0 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 256 --freq_fid 100 --use_subset --trainG --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --cosineLrSched



python trainer_submitit_transfer.py --name ade20kSets_transfer/coco256/classical_finet/trainAll_lrg0.00008_lrd0.0001  --dataset_mode ade20k --gpu_ids 0 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 256 --freq_fid 100 --use_subset --trainG --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --lr_g 0.00008 --lr_d 0.0001

######################################### FREEZE D #########################################
python trainer_submitit_transfer.py --name ade20kSets_transfer/coco256/classical_finet/freezeD_fixG0  --transfer_alt_g 0 --dataset_mode ade20k --gpu_ids 0 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --transfer_alt_d 3 --freezeD_list 6 --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer 

python trainer_submitit_transfer.py --name ade20kSets_transfer/coco256/classical_finet/freezeD_fixG0_cosineLrSched  --transfer_alt_g 0 --dataset_mode ade20k --gpu_ids 0 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --transfer_alt_d 3 --freezeD_list 6 --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer  --cosineLrSched

########################################### WITH CONFMAT #########################################

python trainer_submitit_transfer.py --name ade20kSets_transfer/coco256/classical_finet_wConfMat/trainAll --dataset_mode ade20k --gpu_ids 0 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 256 --freq_fid 100 --use_subset  --trainD --trainG --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --useConfMat --conf_mat conf_mat/target_ade20k_256x256/bin_conf_mat_sourceCOCO_ --slurm

python trainer_submitit_transfer.py --name ade20kSets_transfer/coco256/classical_finet_wConfMat/fixG0_freezeD --dataset_mode ade20k --gpu_ids 0 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 256 --freq_fid 100 --use_subset --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --useConfMat --conf_mat conf_mat/target_ade20k_256x256/bin_conf_mat_sourceCOCO_ --transfer_alt_g 0 --transfer_alt_d 3 --freezeD_list 6 --slurm