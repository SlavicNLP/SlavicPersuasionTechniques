import os
import yaml
import time
import logging
import concurrent.futures
from typing import Tuple, Union, List, Dict
from dotenv import load_dotenv

import google.generativeai as genai
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# =============================================================================
# Environment Setup and Global Constants
# =============================================================================

# Load environment variables from .env file
load_dotenv()
TEMPERATURE = 0.0

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODELS = ["gemini-2.0-flash"]
# =============================================================================
# Helper Functions
# =============================================================================


def ensure_directory_exists(file_path: str) -> None:
    """Ensures that the parent directory for the given file path exists."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def update_dataframe_result(
        df: pd.DataFrame, index: int, system_prompt: str, user_prompt: str, completion: str, result_column: str
) -> None:
    """Updates the dataframe with the given result information."""
    df.at[index, "system_prompt"] = system_prompt
    df.at[index, "user_prompt"] = user_prompt
    df.at[index, result_column] = completion


# =============================================================================
# Setup and I/O Functions
# =============================================================================

def setup_logging(log_filename: str, dataset_file: str, model: str, output_file_path: str) -> None:
    """
    Sets up logging configuration and logs initial script parameters.
    """
    ensure_directory_exists(log_filename)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)]
    )
    logging.info("Script started with the following parameters:")
    logging.info(f"Test file path: {dataset_file}")
    logging.info(f"Model: {model}")
    logging.info(f"Output file path: {output_file_path}")


def read_csv_file(dataset_file: str, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Reads a CSV file into a Pandas DataFrame.
    """
    try:
        df = pd.read_csv(dataset_file, encoding=encoding)
        logging.info(f"Successfully read the file: {dataset_file}, Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error reading CSV file {dataset_file}: {e}", exc_info=True)
        raise


def load_prompts(
        prompts_file_path: str,
        method_type: str
) -> Tuple[str, str]:
    """
    Loads prompts from a YAML file for a given method type.

    Returns:
        A tuple containing the system prompt and user prompt.
    """
    try:
        with open(prompts_file_path, "r", encoding="utf-8") as file:
            prompts = yaml.safe_load(file)

        system_prompt = prompts[method_type].get("system")
        user_prompt = prompts[method_type].get("user")
        if system_prompt is None or user_prompt is None:
            raise KeyError(f"Missing required prompt fields for prompt type: {method_type}")
        return system_prompt, user_prompt
    except (FileNotFoundError, yaml.YAMLError, KeyError, ValueError) as e:
        raise RuntimeError(f"Failed to load prompts: {e}")

# =============================================================================
# API Client Function
# =============================================================================


def client_instance(model: str) -> Union[OpenAI]:
    """
    Returns an appropriate API client instance based on the model string.
    """
    if model in ["gpt-4o-mini", "gpt-4.1-mini"]:
        return OpenAI(api_key=OPENAI_API_KEY)
    elif model in ["meta-llama/Llama-3.3-70B-Instruct", "google/gemma-3-27b-it"]:
        return OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
    elif model in ["gemini-2.0-flash"]:
        return OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    else:
        raise ValueError(f"Unsupported model: {model}")


# =============================================================================
# Processing Functions
# =============================================================================

def process_text_with_model(
        index: int,
        text: str,
        model: str,
        system_prompt: str,
        user_prompt: str
) -> Dict[str, Union[int, str, None]]:
    """
    Processes a single text with a given model.
    Uses Gemini if applicable, otherwise makes a standard chat call.
    Returns a result dictionary.
    """
    try:
        full_prompt = f"{user_prompt} Text:{text}. Answer:"
        if model in GEMINI_MODELS:
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel(
                model_name=model, system_instruction=system_prompt
            )
            response = gemini_model.generate_content(
                full_prompt,
                generation_config=genai.GenerationConfig(temperature=TEMPERATURE)
            )
            completion = response.text
        else:
            client = client_instance(model=model)
            completion_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=TEMPERATURE
            )
            completion = completion_response.choices[0].message.content
        return {
            "index": index,
            "system_prompt": system_prompt,
            "user_prompt": full_prompt,
            "completion": completion,
        }
    except Exception as e:
        print(f"Error processing row {index}: {e}")
        time.sleep(2)  # Brief pause in case of repeated errors
        return {"index": index, "system_prompt": None, "user_prompt": None, "completion": None}


# =============================================================================
# Runner Functions (Parallel)
# =============================================================================

def parallel_text_processing(
        df: pd.DataFrame,
        col_with_content: str,
        result_column: str,
        output_file: str,
        model: str,
        system_prompt: str,
        user_prompt: str
) -> None:
    """
    Parallel processing for the simpler single–prompt workflow.
    """
    df["system_prompt"] = None
    df["user_prompt"] = None
    df[result_column] = None
    ensure_directory_exists(output_file)

    results: List[Dict[str, Union[int, str, None]]] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_text_with_model, idx, text, model, system_prompt, user_prompt)
            for idx, text in enumerate(df[col_with_content])
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Thread failed with error: {e}")

    # Update DataFrame with results
    for result in results:
        update_dataframe_result(df, result["index"], result["system_prompt"], result["user_prompt"],
                                result["completion"], result_column)
    df.to_csv(output_file, index=False)


def process_text(
        df: pd.DataFrame,
        model: str,
        col_with_content: str,
        result_column: str,
        output_file_path: str,
        system_prompt: str,
        user_prompt: str
) -> None:
    """
    Processes text using the appropriate workflow based on the model.
    """
    logging.info("Starting text processing.")
    try:
        parallel_text_processing(
            df.copy(),
            col_with_content=col_with_content,
            result_column=result_column,
            output_file=output_file_path,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        logging.info(f"Processing completed successfully. Results saved to: {output_file_path}")
    except Exception as e:
        logging.error("An error occurred during text processing.", exc_info=True)
