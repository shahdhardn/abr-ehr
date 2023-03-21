srun --ntasks=1 --mem=40G --job-name=ot --cpus-per-task=16 -p gpu -q gpu-8 --gres=gpu:2 --output=OT-batch-20-MBertLstm.out python -W ignore::UserWarning /home/mai.kassem/sources/abr-ehr/OT/OT_train.py
# srun --ntasks=1 --mem=40G --cpus-per-task=16 -p gpu -q gpu-8 --gres=gpu:2 python -W ignore::UserWarning /home/mai.kassem/sources/abr-ehr/OT/OT_train.py --epochs=200 --model="MBertLstm"