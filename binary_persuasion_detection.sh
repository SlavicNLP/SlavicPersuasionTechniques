#!/opt/homebrew/bin/bash

# Define models (comment out the ones you don't want to use)
models=(
    "gpt-4.1-mini"
    "gpt-4o-mini"
    "gemini-2.0-flash"
    "meta-llama/Llama-3.3-70B-Instruct"
    "google/gemma-3-27b-it"
)

# Define datasets (just the language codes)
datasets=("PL" "RU" "BG")

# Prompts file path
prompts_file_path="prompts/binary_persuasion_strategies_detection.yaml"

# Define method types
method_types=(
    "Attack_on_Reputation"
    "Justification"
    "Distraction"
    "Simplification"
    "Call"
    "Manipulative_Wording"
)

# Function to run the script
run_script() {
    local dataset_code=$1
    local model=$2
    local method_type=$3

    local dataset_file="data/${dataset_code}/test.csv"
    local output_file="results/${model}/${dataset_code}/${method_type}/persuasion_binary_detection.csv"

    echo "Processing: $dataset_file for persuasion $method_type using model $model..."

    # Run the Python script
    uv run src/persuasion_classification.py \
        -dataset_file "$dataset_file" \
        -model "$model" \
        -output "$output_file" \
        -prompts_file_path "$prompts_file_path" \
        -method_type "$method_type"
}

# Loop through datasets, models, and method types
for dataset_code in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for method_type in "${method_types[@]}"; do
      run_script "$dataset_code" "$model" "$method_type"
    done
  done
done
