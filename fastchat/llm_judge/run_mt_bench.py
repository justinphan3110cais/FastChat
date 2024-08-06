import fire
import subprocess
import os

def main(model_path, 
        model_id,
        answer_file,
        num_gpus,
        openai_api_key,
        parallel=10,
        num_gpus_per_model=0.5,
        ):
    print("===Generating model Answer ===")

    subprocess.run([
        "python", "gen_model_answer.py", 
        "--num-gpus-per-model", str(1),
        "--num-gpus-total", str(num_gpus),
        "--model-path", model_path,
        "--model-id", model_id,
        "--answer-file", answer_file,
        "--num-gpus-per-model", str(num_gpus_per_model),
    ])

    print("===Generating GPT Score ===")
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = openai_api_key
    subprocess.run([
        "python", "gen_judgment.py", 
        "--model-list", model_id,
        "--parallel", str(parallel)
    ], env=env)

    print("===MT Bench Results ===")
    subprocess.run([
        "python", "show_result.py", 
        "--model-list", model_id,
    ], env=env)


if __name__ == "__main__":
    fire.Fire(main)