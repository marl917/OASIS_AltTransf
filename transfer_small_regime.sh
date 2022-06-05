python trainer_submitit.py --name cityscapes_transfer/20%TrainData_transfercoco_allTrainable_split1 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100 --trainG --trainD --size 512 --partition scavenge --perctg_train_data 0.2 --split 1 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_transfercoco_allTrainable_split2 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100 --trainG --trainD --size 512 --partition scavenge --perctg_train_data 0.2 --split 2 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_transfercoco_allTrainable_split3 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100 --trainG --trainD --size 512 --partition scavenge --perctg_train_data 0.2 --split 3 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_transfercoco_allTrainable_split4 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100 --trainG --trainD --size 512 --partition scavenge --perctg_train_data 0.2 --split 4 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_transfercoco_allTrainable_split5 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/coco/pretrained_coco100 --trainG --trainD --size 512 --partition scavenge --perctg_train_data 0.2 --split 5 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_split5 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --size 512 --partition scavenge --perctg_train_data 0.2 --split 5 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_split4 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --size 512 --partition scavenge --perctg_train_data 0.2 --split 4 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_split3 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --size 512 --partition scavenge --perctg_train_data 0.2 --split 3 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_split2 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --size 512 --partition scavenge --perctg_train_data 0.2 --split 2 --slurm

python trainer_submitit.py --name cityscapes_transfer/20%TrainData_split1 --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --size 512 --partition scavenge --perctg_train_data 0.2 --split 1 --slurm

