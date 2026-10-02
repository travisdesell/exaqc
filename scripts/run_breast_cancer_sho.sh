# Breast cancer runs with the training hyperparameters co-evolved by simplex
# hyperparameter optimization (SHO), using the same search settings as
# run_breast_cancer.sh. Tunes the learning rate and Adam's settings with the
# ranges from Kini et al., GECCO '23, plus the search's crossover rates (starting
# around the configured 0.1 each, free to move anywhere in [0, 1]) and the number
# of mutations per child (starting in 1-3, like -ms uniform 1 3, up to 10).
#
# Usage: bash scripts/run_breast_cancer_sho.sh MIN_COUNT MAX_COUNT OUT_DIR

MIN_COUNT=$1
MAX_COUNT=$2
OUT_DIR=$3
DATASET="breast_cancer"

for i in $(seq $MIN_COUNT $MAX_COUNT); do
    mpiexec --oversubscribe -n 12 python3 -m src.examples.classification --logging_level INFO --dataset $DATASET --number_genomes 1000 --input_qubits 8 --output_qubits 1 --batch_size 3 --mutation_strategy uniform 1 3 --parent_strategy uniform 5 5 --binary_crossover_rate 0.1 --n_ary_crossover_rate 0.1 --exponential_crossover_rate 0.1 -qim ry -qom probs --encoding linear --decoding clipped --hyperparameter_strategy simplex --sho_tune learning_rate=log:1e-3:5e-2:1e-5:0.3 adam_beta1=linear:0.9:0.99 adam_beta2=linear:0.9:0.99 adam_epsilon=log:1e-9:1e-8 binary_crossover_rate=linear:0.05:0.15:0:1 n_ary_crossover_rate=linear:0.05:0.15:0:1 exponential_crossover_rate=linear:0.05:0.15:0:1 mutation_count=int:1:3:1:10 --sho_genomes 4 --sho_l1 2.0 --sho_l2 0.5 --out_dir $OUT_DIR/${DATASET}_i30_sho_${i} steady_state --max_population_size 30
done
