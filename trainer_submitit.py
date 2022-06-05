from train import main
from submitit.helpers import Checkpointable
import submitit
from argparse import ArgumentParser
from config import read_arguments
import argparse
import sys
import os
import config



class Trainer(Checkpointable):
    def __call__(self,args, parser):
     main(args, parser)




if __name__ == "__main__":
    # --- read options ---#
    opt, parser = config.read_arguments_submitit(train=True)

    trainer = Trainer()

    num_gpus = len(opt.gpu_ids.split(','))
    print(f"Using {num_gpus} gpus")
    print(opt.slurm)

    if not opt.slurm:
        trainer(opt, parser)

    else:
        os.makedirs(opt.submLogFolder, exist_ok=True)
        executor = submitit.SlurmExecutor(
            folder=opt.submLogFolder,
            max_num_timeout=60)

        executor.update_parameters(
            gpus_per_node=num_gpus, partition=opt.partition, constraint='volta32gb',
            nodes=1,
            cpus_per_task=num_gpus*10,
            # mem=256000,
            time=4320, job_name=opt.name,
            exclusive=True)
        job = executor.submit(trainer, opt, parser)
        print(job.job_id)

        import time
        time.sleep(1)
