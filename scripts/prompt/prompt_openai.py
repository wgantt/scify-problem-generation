import aiohttp
import asyncio
import click
import json
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from itertools import batched

# from tqdm import tqdm
from tqdm.asyncio import tqdm
from typing import List

# Ensure no one uses a model other than gpt-4o-mini-2024-07-18
GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
O3 = "o3-2025-04-16"
SUPPORTED_MODELS = frozenset({GPT_4O_MINI, O3})
API_KEY = os.environ.get("OPENAI_API_KEY")
CHAT_COMPLETIONS_ENDPOINT = "https://api.openai.com/v1/chat/completions"
headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}


@click.command()
@click.argument("prompt_file", type=str)
@click.argument("output_file", type=str)
@click.option("--model", type=str, default=GPT_4O_MINI)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Max tokens to generate (should be set to the same value as the fine-tuned models)",
)
@click.option(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    help="Number of requests to issue in parallel",
)
@click.option(
    "--seed",
    type=int,
    default=1337,
    help="Random seed. If set to -1, will cycle through a list of seeds",
)
@click.option(
    "--resume",
    is_flag=True,
    help="If true, will filter out examples from prompt_file that are already in output_file",
)
def prompt_all(
    prompt_file, output_file, model, max_tokens, temperature, batch_size, seed, resume
) -> None:
    assert model in SUPPORTED_MODELS, f"Unsupported model: {model}"

    seen_examples = set()
    if resume:
        if not os.path.exists(output_file):
            print(f"Output file {output_file} does not exist. Skipping resume.")
        else:
            with open(output_file, "r") as f:
                for line in f:
                    example = json.loads(line)
                    # instance_id is expected to be a unique identifier for each example
                    seen_examples.add(example["instance_id"])
            print(
                f"Skipping {len(seen_examples)} examples already present in {output_file}."
            )

    # load examples from JSONL-formatted file
    # expected keys are:
    # - instance_id: a unique identifier for each example
    # - user_prompt: a user prompt to be supplied to the model
    # - system_prompt: a system prompt to be supplied to the model
    # - meta (optional): optional metadata
    examples = []
    with open(prompt_file, "r") as f:
        for line in f:
            example = json.loads(line)
            # Skip examples that have already been seen
            if resume and example["instance_id"] in seen_examples:
                continue
            else:
                examples.append(example)

    if resume:
        print(
            f"Loaded {len(examples)} examples from {prompt_file} "
            f"(excluding {len(seen_examples)} already seen examples)."
        )
    else:
        print(f"Loaded {len(examples)} examples from {prompt_file}.")

    # Batch all requests
    for batch in tqdm(
        batched(examples, batch_size),
        desc="Prompting...",
        total=len(examples) // batch_size,
    ):
        results = asyncio.run(prompt_batch(model, batch, max_tokens, temperature, seed))

        # Write results to output
        with open(output_file, "a") as f:
            for e, r in zip(batch, results):
                o = {
                    "instance_id": e["instance_id"],
                    "user_prompt": e["user_prompt"],
                    "system_prompt": e["system_prompt"],
                    "meta": e.get("meta", {}),
                    "response": r,
                }
                f.write(json.dumps(o) + "\n")


async def prompt_batch(model, examples, max_tokens, temperature, seed) -> List[str]:
    # Run prompts
    user_prompts = [ex["user_prompt"] for ex in examples]
    system_prompts = [ex["system_prompt"] for ex in examples]
    async with aiohttp.ClientSession() as session:
        tasks = [
            prompt(session, model, u, max_tokens, temperature, s, seed)
            for (u, s) in zip(user_prompts, system_prompts)
        ]
        return await asyncio.gather(*tasks)


# retry to avoid being rate-limited
@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=2, max=60))
async def prompt(
    session: aiohttp.ClientSession,
    model: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    seed: int,
) -> str:
    # prompt the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    data = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "seed": seed,
    }
    # temperature supported only for non-reasoning models
    if not model.startswith("o3") or model.startswith("o4"):
        data["temperature"] = temperature

    async with session.post(
        CHAT_COMPLETIONS_ENDPOINT, headers=headers, json=data
    ) as response:
        resp = await response.json()
        if "choices" in resp:
            return resp["choices"][0]["message"]["content"]
        elif "error" in resp:
            raise RuntimeError(resp["error"])
        else:
            raise RuntimeError(f"Unexpected response from OpenAI API: {resp}")


if __name__ == "__main__":
    prompt_all()
