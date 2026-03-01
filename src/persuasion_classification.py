import argparse
import logging
import os
from datetime import datetime

# Import all necessary functions from the utils.
from utils.utils import (
    process_text,
    load_prompts,
    setup_logging,
    read_csv_file
)


def configure_logging(model, method_type, dataset_file, output_file_path):
    """
    Configures logging: creates a log directory based on model and method_type and
    creates a log file with a datetime stamp.
    """
    log_dir = os.path.join(output_file_path.split("/")[0], "logging", model, method_type)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(
        log_dir, f"logs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    setup_logging(
        log_filename=log_filename,
        dataset_file=dataset_file,
        model=model,
        output_file_path=output_file_path,
    )


def parse_arguments():
    """
    Parses command-line arguments. The unified script handles both simple detection and Augmented CoT modes.
    """
    parser = argparse.ArgumentParser(
        description="Unified processing script: Choose either a simple detection or an Augmented CoT method."
    )
    parser.add_argument(
        "-dataset_file",
        type=str,
        required=True,
        help="Path to the test dataset file (CSV format).",
    )
    parser.add_argument(
        "-model", type=str, required=True, help="Model name to use for processing."
    )
    parser.add_argument(
        "-output_file_path",
        type=str,
        required=True,
        help="Output file path to save the results.",
    )
    parser.add_argument(
        "-prompts_file_path",
        type=str,
        required=True,
        help="Path to the prompts file (YAML format is also supported).",
    )
    parser.add_argument(
        "-method_type",
        type=str,
        required=True,
        help="Prompting method to use. Options: 'simple_detection', "
             "'icot_one_detailed_multistep', etc.",
    )
    return parser.parse_args()


def simple_detection_branch(args, df):
    """
    Executes the simple detection processing branch.
    Loads prompts and then processes the text using process_text.
    """
    logging.info(f"Executing simple detection branch with method_type: {args.method_type}")
    try:
        logging.info(f"Loading prompts with prompt_type: {args.method_type}")
        system_prompt, user_prompt = load_prompts(
            prompts_file_path=args.prompts_file_path,
            method_type=args.method_type
        )
    except:
        logging.error("Error loading prompts for simple detection.", exc_info=True)
        raise

    try:
        process_text(
            df=df.copy(),
            model=args.model,
            col_with_content="content",
            result_column="final_pred",
            output_file_path=args.output_file_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        logging.info(f"Simple detection processing completed successfully. Results saved to: {args.output_file_path}")
    except:
        logging.error("Error processing text in simple detection branch.", exc_info=True)
        raise


def main():
    args = parse_arguments()

    # Configure logging
    configure_logging(args.model, args.method_type, args.dataset_file, args.output_file_path)

    # Read the dataset
    logging.info("Reading dataset...")
    try:
        df = read_csv_file(args.dataset_file)
    except:
        logging.error("Error reading the dataset file.", exc_info=True)
        raise

    # Decide which branch to execute based on the method_type.
    simple_detection_branch(args, df)


if __name__ == "__main__":
    main()
