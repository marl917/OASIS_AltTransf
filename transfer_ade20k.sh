python trainer_submitit.py --name cityscapes_transfer/100%TrainData_transferAde20k_allTrainable --dataset_mode cityscapes --gpu_ids 0,1,2,3 --dataroot /datasets01/cityscapes/112817 --batch_size 20 --num_epochs 5000 --transfer_l /checkpoint/marlenec/oasis_training/ade20k/pretrained_models_100 --trainG --trainD --size 512 --partition learnlab --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transferAde20k


python trainer_submitit.py --name facades/100%TrainData_transferAde20k_allTrainable --dataset_mode facades --gpu_ids 0,1,2,3 --dataroot /datasets01/ADE20kChallengeData2016/011719/ --batch_size 32 --num_epochs 50000 --transfer_l /checkpoint/marlenec/oasis_training/ade20k/pretrained_models_100 --trainG --trainD --slurm --submLogFolder /checkpoint/marlenec/oasis_submitit_logs_transferAde20k