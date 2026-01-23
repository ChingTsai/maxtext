import argparse
import transformers
import datasets
import grain
from tqdm import tqdm
import os
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

def process(entry, tokenizer, messages_col="messages"):
    assert entry["messages"][-1]["role"] == "assistant"
    entry["prompt"] = tokenizer.apply_chat_template(
        entry[messages_col][:-1],
        tokenize=False,
        add_generation_prompt=True
    )
    entry["target_answer"] = entry["messages"][-1]["content"]
    return entry

def score_response_chid(target, prediction):
  """Scores the model's prediction against the target answer for CHID."""
  letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  key = "答案是"

  def extract_answer(text):
    key_idx = text.find(key)
    if key_idx == -1:
      return ""
    ans_str = text[key_idx + len(key) :].strip()
    tmp_s = ""
    for char in ans_str:
     if char in letters:
       tmp_s += char
    return tmp_s
  def calculate_match_counts(pred: str, answer: str):
    match_count = sum(1 for p, a in zip(pred, answer) if p == a)
    total_count = len(answer)
    
    return match_count, total_count
  
  pred_ans = extract_answer(prediction)
  target_ans = extract_answer(target)
  msg = (f"Compare parsed predict: {pred_ans} with target: {target_ans}")
  match_count, total_count = calculate_match_counts(pred_ans, target_ans)

  return match_count, total_count, msg

def evaluate_model_chid_native(dataset, llm, sp,debug=True):
  """Runs evaluation on the model for CHID dataset using vLLM.
  Args:
    dataset: The dataset to evaluate on, with 'prompt' and 'target_answer'.
    vllm_rollout: The vLLM rollout object for generating responses.
    debug: If True, prints debug information for each sample.

  Returns:
    A dictionary containing evaluation score: 'accuracy' percentage.
  """
  total, total_correct = 0, 0
  for batch in tqdm(dataset):
    batch_response = llm.generate(batch["prompt"], sp)
    for i, prompt in enumerate(batch["prompt"]):
      gen_text = batch_response[i].outputs[0].text
      
      pred_correct, all_correct , msg = score_response_chid(target=batch["target_answer"][i], prediction=gen_text)
      if debug:
        print("========================================")
        print(f"Prompt: {prompt}")
        print("----------------------------------------")
        print(f"Model Generated Response: \n{batch_response.text[i]}")
        print("----------------------------------------")
        print(f"Target Response: {batch['target_answer'][i]}")
        print("========================================")
        print(msg)
      total += all_correct
      total_correct += pred_correct
  return {
      "total": total,
      "total_correct": total_correct,
      "accuracy": (total_correct / total) * 100,
  }

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for Qwen models")
    parser.add_argument(
        "--chk_path", 
        type=str, 
        default="/mnt/disks/jimmy_workspace/jimmy-bs16-gs1-lr7e-6/",
        help="Path to the model checkpoint directory"
    )
    return parser.parse_args()


def main():
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    args = parse_args()
    chk_path = args.chk_path
    
    print(f"Starting evaluation for checkpoint: {chk_path}")

    model_name = "Qwen/Qwen2.5-14B-Instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(chk_path)
    
    hdataset = datasets.load_dataset("arrow",data_files="gs://jimmytsai-dev/tencent_sft/clid-data-arrow/data-sys-ep2s1/validation/data-00000-of-00001.arrow", split="train").map(lambda x : process(x ,tokenizer))
    
    gdataset = (
          grain.MapDataset.source(hdataset)
      )
    
    llm = LLM(model=chk_path, dtype='bfloat16',max_model_len=1024)
    
    tmps = [0.5] 
    repetition_penalties = [1.1]
    for tmp in tmps:
        for repetition_penalty in repetition_penalties:
            sp = SamplingParams(temperature=tmp, max_tokens=128, repetition_penalty=repetition_penalty)
            dataset = gdataset.batch(400, drop_remainder=True)
            print(f"temp: {tmp}, repetition_penalty: {repetition_penalty}")
            print(evaluate_model_chid_native(dataset, llm, sp, debug=False))

if __name__ == "__main__":
    main()