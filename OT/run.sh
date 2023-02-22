# srun --ntasks=1 --mem=256G --cpus-per-task=16 -p gpu -q gpu-8 --gres=gpu:4 --output=OT-batch-40.out python -W ignore::UserWarning /home/mai.kassem/sources/abr-ehr/OT/OT_train.py
srun --ntasks=1 --mem=64G --cpus-per-task=16 -p gpu -q gpu-8 --gres=gpu:1 --output=OT-batch-16.out python -W ignore::UserWarning /home/mai.kassem/sources/abr-ehr/OT/OT_train.py --model="MBertLstm"
