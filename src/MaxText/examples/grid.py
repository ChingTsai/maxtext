from absl import app
import MaxText
import dotenv
import os
from datetime import datetime
dotenv.load_dotenv(override=True)
HF_TOKEN = os.environ.get("HF_TOKEN")
MAXTEXT_REPO_ROOT = os.path.dirname(MaxText.__file__)

from tqdm.auto import tqdm
from typing import Sequence

import datasets
import grain
import os
import re
import transformers

from flax import nnx

from MaxText import max_logging
from MaxText import max_utils
from MaxText import pyconfig
from MaxText.integration.tunix.tunix_adapter import TunixMaxTextAdapter
from MaxText.sft import sft_trainer
# Suppress vLLM logging with a severity level below ERROR
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout.vllm_rollout import VllmRollout

# Skip JAX precompilation to make vLLM startup faster
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

SEED = 42
BATCH_SIZE = 20
NUM_TEST_SAMPLES = 100
MAX_TOKENS_TO_GENERATE = 128
MAX_PROMPT_LENGTH = 1000
EVALUATION_CONFIG = {
  "temperature": 0.5, 
  #"repetition_penalty": 1.1
  }


def get_test_dataset(config, tokenizer):
  """Loads and prepares the test dataset from Hugging Face.

  Args:
    config: The pyconfig object containing run configurations, including
      `hf_access_token`.
    tokenizer: The tokenizer for processing the text data.

  Returns:
    A grain.MapDataset instance for the test split, with prompts and target
    answers.
  """
  def process(entry, tokenizer, messages_col="messages"):
    assert entry["messages"][-1]["role"] == "assistant"
    #entry["messages"][-2]["content"]+=" /no_think"
    entry["prompt"] = tokenizer.apply_chat_template(
        entry[messages_col][:-1],
        tokenize=False,
        add_generation_prompt=True
    )
    entry["target_answer"] = entry["messages"][-1]["content"]
    return entry
  
  dataset = datasets.load_dataset(
      config.hf_path,
      data_dir=config.hf_data_dir,
      split=config.hf_eval_split,
      token=config.hf_access_token,
  ).map(lambda x : process(x,tokenizer))

  return (
      grain.MapDataset.source(dataset)
      .shuffle(seed=SEED)
  )


def evaluate_model_chid(dataset, vllm_rollout, debug=True):
  """Runs evaluation on the model for CHID dataset using vLLM.
  Args:
    dataset: The dataset to evaluate on, with 'prompt' and 'target_answer'.
    vllm_rollout: The vLLM rollout object for generating responses.
    debug: If True, prints debug information for each sample.

  Returns:
    A dictionary containing evaluation score: 'accuracy' percentage.
  """
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_TOKENS_TO_GENERATE,
      max_prompt_length=MAX_PROMPT_LENGTH,
      data_type="bfloat16",
      **EVALUATION_CONFIG
  )

  total, total_correct = 0, 0
  for batch in tqdm(dataset):
    batch_response = vllm_rollout.generate(batch["prompt"], rollout_config)
    for i, prompt in enumerate(batch["prompt"]):
      
      is_correct, msg = score_response_chid(target=batch["target_answer"][i], prediction=batch_response.text[i])
      if is_correct:
        total_correct += 1
      else:
        if debug:
          print("========================================")
          print(f"Prompt: {prompt}")
          print("----------------------------------------")
          print(f"Model Generated Response: \n{batch_response.text[i]}")
          print("----------------------------------------")
          print(f"Target Response: {batch['target_answer'][i]}")
          print("========================================")
          print(msg)
      total += 1

  return {
      "accuracy": (total_correct / total) * 100,
  }


def score_response_chid(target, prediction):
  """Scores the model's prediction against the target answer for CHID."""
  letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  key = "答案是"

  def extract_answer(text):
    key_idx = text.find(key)
    if key_idx == -1:
      return ""
    ans_str = text[key_idx + len(key) :].strip()
    return ans_str
    #for char in ans_str:
    #  if char in letters:
    #    return char
    #return ""

  pred_ans = extract_answer(prediction)
  target_ans = extract_answer(target)
  msg = (f"Compare parsed predict: {pred_ans} with target: {target_ans}")
  return pred_ans == target_ans, msg


def create_vllm_rollout(config, model, mesh, tokenizer):
  """Creates a vLLM rollout engine for text generation.

  Args:
    config: The pyconfig object containing run configurations.
    model: The NNX model graph.
    mesh: The JAX device mesh.
    tokenizer: The tokenizer.

  Returns:
    A VllmRollout instance configured for the model and hardware.
  """
  tunix_model = TunixMaxTextAdapter(model)
  return VllmRollout(
      model=tunix_model,
      tokenizer=tokenizer,
      cache_config_or_size=MAX_PROMPT_LENGTH + MAX_TOKENS_TO_GENERATE + 256,
      mesh=mesh,
      model_version=config.tokenizer_path,
      hbm_utilization=0.8,
      init_with_random_weights=True,
      tpu_backend_type="jax",
  )


def get_tokenizer(config):
  """Initializes and returns the tokenizer.

  Args:
    config: The pyconfig object with `tokenizer_path` and `hf_access_token`.

  Returns:
    A Hugging Face tokenizer instance.
  """
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.tokenizer_path,
      token=config.hf_access_token,
  )
  return tokenizer


def main(argv: Sequence[str]) -> None:
  """Main function to run SFT training and evaluation.

  Args:
    argv: Command-line arguments.
  """
  MODEL_NAME='qwen2.5-14b'
  TOKENIZER_PATH="Qwen/Qwen2.5-14B-Instruct"
  DATASET_NAME="/home/jimmytsai_google_com/workspace/tencent_sft/clid-data"
  RUN_NAME=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

  TRAIN_SPLIT="train"
  HF_EVAL_SPLIT="validation"
  HF_DATA_DIR="data"
  TRAIN_STEPS=1000
  CONVERTED_CKPT_DIR="/mnt/disks/jimmy_workspace/qwen2.5-14b_checkpoint"
  MODEL_CHECKPOINT_PATH=f"{CONVERTED_CKPT_DIR}/0/items"
  #MODEL_CHECKPOINT_PATH="/home/jimmytsai_google_com/workspace/maxtext_qwen2.5-14b/2025-12-30-10-03-58/checkpoints/226/model_params"
  OUTPUT_PATH=f"/mnt/disks/jimmy_workspace/maxtext_{MODEL_NAME}"

  common_argv_dict = {
      #"run_name":RUN_NAME,
      "model_name": MODEL_NAME,
      "load_parameters_path":MODEL_CHECKPOINT_PATH,
      "base_output_directory":OUTPUT_PATH,
      "hf_access_token": HF_TOKEN,
      "tokenizer_path": TOKENIZER_PATH,
      "tokenizer_type": "huggingface",
      "hf_path": DATASET_NAME,
      "train_split": TRAIN_SPLIT,
      #"hf_data_dir": HF_DATA_DIR,
      "hf_eval_split": HF_EVAL_SPLIT,
      "train_data_columns": ["messages"],
      #"per_device_batch_size": 1,
      "steps": TRAIN_STEPS,
      "dtype": "bfloat16",
      "num_epoch": 1,
      "weight_dtype": "bfloat16",
      #"learning_rate": 5e-6,
      "warmup_steps_fraction": 0.05,
      "max_target_length": 1024,
      "eval_steps": 10,
      "eval_interval": 100,
      "skip_first_n_steps_for_profiler": 100,  
  }
  
  for key, value in common_argv_dict.items():
    argv.append(f"{key}={value}")

  config = pyconfig.initialize(argv)
  max_utils.print_system_information()

  trainer, mesh = sft_trainer.setup_trainer_state(config)
  trainer = sft_trainer.train_model(config, trainer, mesh)

  tokenizer = get_tokenizer(config)
  vllm_rollout = create_vllm_rollout(config, trainer.model, mesh, tokenizer)

  tokenizer = get_tokenizer(config)
  test_dataset = get_test_dataset(config, tokenizer)
  test_dataset = test_dataset
  test_dataset = test_dataset.to_iter_dataset().batch(20, drop_remainder=True)

  max_logging.log("Running Pre-SFT evaluation...")
  score = evaluate_model_chid(test_dataset, vllm_rollout)
  print("Score for PRE-SFT EVALUATION: ", score)


if __name__ == "__main__":
  app.run(main)