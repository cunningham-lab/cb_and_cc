sbatch gpu_run.sh 'train_dqn.py --ENV_NAME=SpaceInvadersDeterministic-v4'
sbatch gpu_run.sh 'train_dqn.py --ENV_NAME=AtlantisDeterministic-v4'
sbatch gpu_run.sh 'train_dqn.py --ENV_NAME=EnduroDeterministic-v4'
sbatch gpu_run.sh 'train_dqn.py --ENV_NAME=BreakoutDeterministic-v4'
sbatch gpu_run.sh 'train_dqn.py --ENV_NAME=PongDeterministic-v4'
sbatch gpu_run.sh 'train_dqn.py --ENV_NAME=BoxingDeterministic-v4'
sbatch gpu_run.sh 'train_dqn.py --ENV_NAME=SeaquestDeterministic-v4'
sbatch gpu_run.sh 'train_dqn.py --ENV_NAME=CrazyClimberDeterministic-v4'

