# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example: Get prompt (template) token IDs and completion token IDs from vLLM.

This example shows how to retrieve both the prompt token IDs (the tokenized
template/input) and the completion token IDs (the generated output) when
making requests to vLLM's OpenAI-compatible API.

Two approaches are demonstrated:
  1. Using the `openai` Python client with `extra_body`
  2. Using the `requests` library directly

Usage:
    # Start vLLM server first:
    #   vllm serve Qwen/Qwen2.5-1.5B-Instruct
    python openai_return_token_ids_client.py
    python openai_return_token_ids_client.py --stream
    python openai_return_token_ids_client.py --use-requests
"""

import argparse
import json

import requests
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example: return prompt and completion token IDs"
    )
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    parser.add_argument(
        "--use-requests",
        action="store_true",
        help="Use the requests library instead of the openai client",
    )
    parser.add_argument(
        "--api-base",
        default=openai_api_base,
        help="vLLM API base URL (default: %(default)s)",
    )
    return parser.parse_args()


# ── Approach 1: openai client ────────────────────────────────────────────────


def demo_completion_openai(client, model: str, stream: bool):
    """Completion API with the openai client."""
    print("=" * 60)
    print("Completion API (openai client)")
    print("=" * 60)

    completion = client.completions.create(
        model=model,
        prompt="The capital of France is",
        max_tokens=20,
        temperature=0,
        extra_body={"return_token_ids": True},
        stream=stream,
    )

    if stream:
        first = True
        all_token_ids = []
        for chunk in completion:
            choice = chunk.choices[0]
            if first and choice.prompt_token_ids:
                print(f"  prompt_token_ids: {choice.prompt_token_ids}")
                first = False
            if choice.token_ids:
                all_token_ids.extend(choice.token_ids)
                print(f"  chunk token_ids: {choice.token_ids}  text: {choice.text!r}")
        print(f"  all completion token_ids: {all_token_ids}")
    else:
        choice = completion.choices[0]
        print(f"  prompt_token_ids: {choice.prompt_token_ids}")
        print(f"  completion token_ids: {choice.token_ids}")
        print(f"  text: {choice.text!r}")
    print()


def demo_chat_openai(client, model: str, stream: bool):
    """Chat Completion API with the openai client."""
    print("=" * 60)
    print("Chat Completion API (openai client)")
    print("=" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in three languages."},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=60,
        temperature=0,
        extra_body={"return_token_ids": True},
        stream=stream,
    )

    if stream:
        first = True
        all_token_ids = []
        for chunk in response:
            if first and chunk.prompt_token_ids:
                print(f"  prompt_token_ids: {chunk.prompt_token_ids}")
                first = False
            if chunk.choices and chunk.choices[0].token_ids:
                all_token_ids.extend(chunk.choices[0].token_ids)
        print(f"  all completion token_ids: {all_token_ids}")
    else:
        print(f"  prompt_token_ids: {response.prompt_token_ids}")
        print(f"  completion token_ids: {response.choices[0].token_ids}")
        print(f"  content: {response.choices[0].message.content!r}")
    print()


# ── Approach 2: requests library ─────────────────────────────────────────────


def demo_completion_requests(api_base: str, model: str, stream: bool):
    """Completion API with the requests library."""
    print("=" * 60)
    print("Completion API (requests library)")
    print("=" * 60)

    payload = {
        "model": model,
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0,
        "return_token_ids": True,
        "stream": stream,
    }

    if stream:
        resp = requests.post(
            f"{api_base}/completions", json=payload, stream=True
        )
        resp.raise_for_status()
        first = True
        all_token_ids = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            choice = chunk["choices"][0]
            if first and choice.get("prompt_token_ids"):
                print(f"  prompt_token_ids: {choice['prompt_token_ids']}")
                first = False
            if choice.get("token_ids"):
                all_token_ids.extend(choice["token_ids"])
                print(
                    f"  chunk token_ids: {choice['token_ids']}  "
                    f"text: {choice['text']!r}"
                )
        print(f"  all completion token_ids: {all_token_ids}")
    else:
        resp = requests.post(f"{api_base}/completions", json=payload)
        resp.raise_for_status()
        result = resp.json()
        choice = result["choices"][0]
        print(f"  prompt_token_ids: {choice['prompt_token_ids']}")
        print(f"  completion token_ids: {choice['token_ids']}")
        print(f"  text: {choice['text']!r}")
    print()


def demo_chat_requests(api_base: str, model: str, stream: bool):
    """Chat Completion API with the requests library."""
    print("=" * 60)
    print("Chat Completion API (requests library)")
    print("=" * 60)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in three languages."},
        ],
        "max_tokens": 60,
        "temperature": 0,
        "return_token_ids": True,
        "stream": stream,
    }

    if stream:
        resp = requests.post(
            f"{api_base}/chat/completions", json=payload, stream=True
        )
        resp.raise_for_status()
        first = True
        all_token_ids = []
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            if first and chunk.get("prompt_token_ids"):
                print(f"  prompt_token_ids: {chunk['prompt_token_ids']}")
                first = False
            if chunk["choices"] and chunk["choices"][0].get("token_ids"):
                all_token_ids.extend(chunk["choices"][0]["token_ids"])
        print(f"  all completion token_ids: {all_token_ids}")
    else:
        resp = requests.post(f"{api_base}/chat/completions", json=payload)
        resp.raise_for_status()
        result = resp.json()
        print(f"  prompt_token_ids: {result['prompt_token_ids']}")
        print(f"  completion token_ids: {result['choices'][0]['token_ids']}")
        print(f"  content: {result['choices'][0]['message']['content']!r}")
    print()


def main():
    args = parse_args()

    if args.use_requests:
        # Fetch model name via requests
        resp = requests.get(f"{args.api_base}/models")
        resp.raise_for_status()
        model = resp.json()["data"][0]["id"]

        demo_completion_requests(args.api_base, model, args.stream)
        demo_chat_requests(args.api_base, model, args.stream)
    else:
        client = OpenAI(api_key=openai_api_key, base_url=args.api_base)
        model = client.models.list().data[0].id

        demo_completion_openai(client, model, args.stream)
        demo_chat_openai(client, model, args.stream)


if __name__ == "__main__":
    main()
