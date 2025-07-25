import argparse
import asyncio
import json
import os

from anthropic import AsyncAnthropic
from tqdm import tqdm
from typing import List, Dict, Any

DEFAULT_MODEL = "claude-sonnet-4-20250514"
OPUS = "claude-opus-4-20250514"
MAX_TOKENS = 50000


class AsyncClaudeClient:
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)

    async def send_message(
        self, message: str, model: str = DEFAULT_MODEL
    ) -> Dict[str, Any]:
        """Send a single message to Claude API"""
        try:
            response = await self.client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": message}],
            )

            return {
                "success": True,
                "message": message,
                "response": response.content[0].text,
            }

        except Exception as e:
            return {"success": False, "message": message, "error": str(e)}

    async def send_multiple_messages(
        self, messages: List[str], model: str = DEFAULT_MODEL
    ) -> List[Dict[str, Any]]:
        """Send multiple messages concurrently"""
        tasks = [self.send_message(msg, model) for msg in messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during gathering
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {"success": False, "message": messages[i], "error": str(result)}
                )
            else:
                processed_results.append(result)

        return processed_results

    async def send_message_with_system_prompt(
        self, message: str, system_prompt: str, model: str = DEFAULT_MODEL
    ) -> Dict[str, Any]:
        """Send a message with a custom system prompt"""
        async with self.client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": message}],
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    print(event.text, end="", flush=True)
                elif event.type == "content_block_stop":
                    print(
                        "\n\ncontent block finished accumulating:", event.content_block
                    )
            print()

        response = await stream.get_final_message()

        return {
            "success": True,
            "message": message,
            "response": response.content[0].text,
        }

    async def close(self):
        """Close the client connection"""
        await self.client.close()


async def main(args: argparse.Namespace):
    API_KEY = os.getenv("ANTHROPIC_API_KEY")

    client = AsyncClaudeClient(API_KEY)
    model = OPUS if args.opus else DEFAULT_MODEL

    with open(args.prompts_file, "r") as f:
        prompts = [json.loads(line) for line in f if line.strip()]

    try:
        for prompt in tqdm(prompts, desc="Prompting..."):
            result = await client.send_message_with_system_prompt(
                prompt["user_prompt"], prompt["system_prompt"], model=model
            )

            if not result["success"]:
                print(f"Error: {result['error']}")
            else:
                response = json.loads(result["response"])
                response_obj = {
                    "response": response,
                    "user_prompt": prompt["user_prompt"],
                    "system_prompt": prompt["system_prompt"],
                    "meta": prompt["meta"] | {"model": model},
                }
                with open(args.output_file, "a") as out_f:
                    out_f.write(json.dumps(response_obj) + "\n")

    finally:
        # Clean up
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async Claude Client Example")
    parser.add_argument(
        "prompts_file", type=str, help="Path to the file containing prompts"
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output file for responses"
    )
    parser.add_argument(
        "--opus", action="store_true", help="Use Opus model instead of default"
    )
    args = parser.parse_args()

    # Run the main example
    asyncio.run(main(args))

    # Or run the context manager example
    # asyncio.run(main_with_context_manager())
