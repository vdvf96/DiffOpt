#!/bin/bash
#SBATCH --partition="standard"
#SBATCH -A raiselab

# Get the path of the Conda environment
conda_env_name=new_GP_env
conda_env_path=/home/eda8pc/.conda/envs/$conda_env_name
#Read values from config.txt and process each line
input_file="hyperparams_sampling_long_epochs.txt" #"hyperparams_local.txt"

while read -r a b c d e f g h i j k l; do
    output_file="out/output_${l}.out" #_${c}_${d}_${e}_${f}_${g}_${h}_${i}_${j}_${k}.out"
    #echo "Value of a: $a"
    #echo "Value of b: $b"
    # echo "Value of c: $c"
    # echo "Value of d: $d"
    # echo "Value of e: $e"
    # echo "Value of f: $f"
    # echo "Value of g: $g"
    # echo "Value of h: $h"
    # echo "Value of i: $i"
    # Create a separate submission script for each job
    submission_script="submit/submit_${l}.sh"
    cat > "$submission_script" <<EOT
#!/bin/bash
#SBATCH --job-name=submit/myjob_${l}
#SBATCH --error=error/myjob_${l}.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -A raiselab
#SBATCH --time=1-12
#SBATCH --partition="standard"

# Add Conda environment paths to PATH and LD_LIBRARY_PATH
# export PATH="$conda_env_path/bin:\$PATH"
# export LD_LIBRARY_PATH="$conda_env_path/lib:\$LD_LIBRARY_PATH"


################################################################################################ 
#REMOVE EPOCHS AND MAXOUTERITER PARAMS
################################################################################################



# Run your Python script
# $conda_env_path/bin/python3 local_GP_stable_dyn_uniform_sampling.py --group_index $a --gen_index $b --n_interval $c --kernel $d --length_scale $e --other_param $f > "$output_file"

#$conda_env_path/bin/python3 local_GP_stable_dyn_non_uniform_sampling.py --group_index $a --gen_index $b --n_interval $c --kernel $d --length_scale $e --other_param $f > "$output_file"

$conda_env_path/bin/python3 computeKL.py --n_layers $a --hidden_units $b --optimizer $c --conditioning_type $d --activation $e --batch_size $f --lr $g --seed $h  --id $i --eps $j --annealed_step $k --id_script $l > "$output_file" 

#--gen_index $b --n_interval $c --kernel_type $d --kernel2_type $e --lambdA $f --id  $g > "$output_file"

# --group_index $a --gen_index $b --n_interval $c --kernel $d --length_scale $e --other_param $f > "$output_file"

#$conda_env_path/bin/python3 pdl.py --acopf_feature_mapping_type "$a" --probtype "$b" > "$output_file"
# Deactivate the environment

EOT
    
    # Make the submission script executable
    chmod +x "$submission_script"
    
    # Submit the job using the created submission script
    sbatch "$submission_script"
    
    # Optionally, remove the submission script
    rm "$submission_script"

done < "$input_file"
