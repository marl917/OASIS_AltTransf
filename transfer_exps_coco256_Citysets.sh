## to compute confusion matrix
# for i in {0..9}
# do
# 	python test.py  --name coco/100%TrainData --dataset_mode cityscapes --gpu_ids 0,1 --dataroot /datasets01/cityscapes/112817 --batch_size 10 --size 512 --disc_out --subset_nb $i --use_subset --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer
# done

############################################# Exps when finetuning all #########################################
# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/trainAll  --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --trainG --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer

# for lr in 0.0001 0.0008
# do
# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/trainAll_lr$lr  --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --trainG --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --lr_g $lr --lr_d $lr
# done

# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/trainAll_diffLR  --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --trainG --trainD --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --diff_lr_layer --slurm

# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/trainAll_cosineLrSched  --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --trainG --trainD --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --cosineLrSched --slurm


#################### 5/06/2022
for lr in 0.00008 0.00004
do
python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/trainAll_lr$lr  --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --trainG --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --lr_g $lr --lr_d $lr
done



############################################ trainD fixG #########################################
# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/trainD_fixG0  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset  --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer


######################################### FREEZE D #########################################
# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/freezeD_fixG0  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --transfer_alt_d 3 --freezeD_list 4,5,6 --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer 

# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/freezeD_trainG  --trainG --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --transfer_alt_d 3 --freezeD_list 6 --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer 

#################### 5/06/2022
python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/freezeD_fixG0  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --transfer_alt_d 3 --freezeD_list 7,8 --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer 

python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/freezeD_fixG0_cosineLrSched  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --transfer_alt_d 3 --freezeD_list 6 --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --cosineLrSched

python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/freezeD_fixG0_lr0.0001  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --transfer_alt_d 3 --freezeD_list 6 --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --cosineLrSched --lr_g 0.0001 --lr_d 0.0001



########################################### WITH CONFMAT #########################################
# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet_wConfMat/trainD_fixG0  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset  --trainD --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --useConfMat --conf_mat conf_mat/target_cityscapes_256x512/bin_conf_mat_sourceCOCO_ --slurm

# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet_wConfMat/trainAll --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset  --trainD --trainG --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --useConfMat --conf_mat conf_mat/target_cityscapes_256x512/bin_conf_mat_sourceCOCO_ --slurm



#################### 5/06/2022
python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet_wConfMat/fixG0_freezeD  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset  --trainD --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --useConfMat --conf_mat conf_mat/target_cityscapes_256x512/bin_conf_mat_sourceCOCO_ --slurm --transfer_alt_d 3 --freezeD_list 6

python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet_wConfMat/fixG0_freezeD_lr0.0001  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset  --trainD --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --useConfMat --conf_mat conf_mat/target_cityscapes_256x512/bin_conf_mat_sourceCOCO_ --slurm --transfer_alt_d 3 --freezeD_list 6 --lr_g 0.0001 --lr_d 0.0001



############################################## TRAIN ALL ###########################################
python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet/trainAll_batch10  --dataset_mode cityscapes --gpu_ids 0,1 --dataroot /datasets01/cityscapes/112817 --batch_size 10 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --trainG --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer

# python trainer_submitit_transfer.py --name cityscapes_transfer/coco256/classical_finet_wConfMat/trainD_fixG0  --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100/ --size 512 --freq_fid 100 --use_subset --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transfer --useConfMat --conf_mat conf_mat/target_cityscapes_256x512/bin_conf_mat_sourceCOCO_ --slurm --transfer_alt_d 3 --freezeD_list 4,5,6


# python trainer_submitit.py --name cityscapes_transfer/coco256/classical_finet/ --transfer_alt_g 0 --dataset_mode cityscapes --gpu_ids 0 --dataroot /datasets01/cityscapes/112817 --batch_size 6 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/ --size 256 --conf_mat bin_conf_mat_sourceAde20K128cityscapes_fewShottest_subset_Nb$i.pt --freq_fid 500 --subset_nb $i --use_subset --slurm --transfer_alt_d 3 --freezeD 6 --useConfMat