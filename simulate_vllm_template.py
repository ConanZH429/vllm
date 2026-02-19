"""
模拟 vLLM 从收到 requests 请求到应用 chat template 的完整流程。

用法:
    pip install transformers
    python simulate_vllm_template.py

不需要启动 vLLM 服务，也不需要 GPU。
"""

import json
import copy
from transformers import AutoTokenizer


# ============================================================
# 第 1 步：定义请求数据（和你用 requests 发送的 payload 一致）
# ============================================================

tools = [{
    "type": "function",
    "function": {
        "name": "read",
        "description": "Reads a file from the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "description": "The line number to start reading from (0-based)"
                },
                "limit": {
                    "type": "integer",
                    "default": 2000,
                    "description": "The number of lines to read (defaults to 2000)"
                }
            },
            "required": ["filePath"]
        }
    }
}]

messages = [
    {
        "role": "system",
        "content": "你是一个人工智能助手"
    },
    {
        "role": "user",
        "content": "判断src.txt和target.txt文件中是否都包含了'今天是星期天'"
               "这句话，这两个文件都在当前路径，可以直接读取"
    },
    {
        "role": "assistant",
        "reasoning": "先读取src.txt文件的内容",
        "content": "",
        "tool_calls": [
            {
                "id": "call_xK9mN3pL2qR8vT5wY6hZ1aB4",
                "type": "function",
                "function": {
                    "name": "read",
                    "arguments": "{\"filePath\": \"./src.txt\"}"
                }
            }
        ]
    }
]

MODEL_NAME = "Qwen/Qwen3-30B-A3B"


# ============================================================
# 第 2 步：模拟 vLLM 的 parse_chat_messages + _postprocess_messages
#
# 对应源码:
#   vllm/entrypoints/chat_utils.py :: parse_chat_messages()
#   vllm/entrypoints/chat_utils.py :: _parse_chat_message_content()
#   vllm/entrypoints/chat_utils.py :: _postprocess_messages()
# ============================================================

def parse_messages_like_vllm(messages):
    """
    模拟 vLLM 对 messages 的处理:
    1. 提取 reasoning/reasoning_content 字段（interleaved thinking 支持）
    2. 复制 tool_calls、tool_call_id、name 字段
    3. 将 tool_calls 中的 arguments 从 JSON 字符串解析为 dict
    """
    conversation = []

    for message in messages:
        msg = copy.deepcopy(message)
        role = msg["role"]
        content = msg.get("content")

        # 构造 ConversationMessage（对应 _parse_chat_message_content_parts）
        conv_msg = {"role": role, "content": content if content else ""}

        if role == "assistant":
            # 复制 tool_calls（对应 _parse_chat_message_content 中的 assistant 分支）
            if "tool_calls" in msg and msg["tool_calls"] is not None:
                conv_msg["tool_calls"] = list(msg["tool_calls"])

            # 读取 reasoning 或 reasoning_content（interleaved thinking）
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            if reasoning is not None:
                conv_msg["reasoning"] = reasoning
                conv_msg["reasoning_content"] = reasoning

        elif role == "tool":
            if "tool_call_id" in msg:
                conv_msg["tool_call_id"] = msg["tool_call_id"]

        if "name" in msg and isinstance(msg["name"], str):
            conv_msg["name"] = msg["name"]

        conversation.append(conv_msg)

    # 对应 _postprocess_messages：将 arguments 从 JSON 字符串解析为 dict
    for conv_msg in conversation:
        if conv_msg["role"] == "assistant" and "tool_calls" in conv_msg:
            tool_calls = conv_msg.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            if len(tool_calls) == 0:
                conv_msg.pop("tool_calls", None)
                continue
            for item in tool_calls:
                args = item["function"].get("arguments")
                if args:
                    if not isinstance(args, (dict, list)):
                        item["function"]["arguments"] = json.loads(args)
                else:
                    item["function"]["arguments"] = {}

    return conversation


# ============================================================
# 第 3 步：加载 tokenizer 并应用 chat template
#
# 对应源码:
#   vllm/renderers/hf.py :: safe_apply_chat_template()
#     -> tokenizer.apply_chat_template(conversation, tools, ...)
# ============================================================

def apply_template(tokenizer, conversation, tools, enable_thinking=True):
    """
    模拟 vLLM 的 safe_apply_chat_template:
    调用 tokenizer.apply_chat_template 并返回渲染后的字符串。
    """
    tool_dicts = [t.get("function", t) if "function" in t else t for t in tools]

    rendered = tokenizer.apply_chat_template(
        conversation=conversation,
        tools=tool_dicts,
        chat_template=None,          # 使用 tokenizer 自带的模板
        tokenize=False,              # 返回字符串而非 token ids
        add_generation_prompt=True,  # vLLM chat completion 默认为 True
        enable_thinking=enable_thinking,
    )
    return rendered


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"=== 加载 tokenizer: {MODEL_NAME} ===\n")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 第 2 步：解析消息
    print("=== 第 2 步：parse_chat_messages (模拟 vLLM 消息解析) ===\n")
    conversation = parse_messages_like_vllm(messages)

    print("解析后的 conversation:")
    for i, msg in enumerate(conversation):
        print(f"  [{i}] role={msg['role']}, keys={list(msg.keys())}")
        if "reasoning" in msg:
            print(f"      reasoning={msg['reasoning']!r}")
    print()

    # 第 3 步：应用模板
    print("=== 第 3 步：apply_chat_template (模拟 vLLM 模板渲染) ===\n")
    rendered = apply_template(tokenizer, conversation, tools, enable_thinking=True)

    print("渲染结果:")
    print("-" * 60)
    print(rendered)
    print("-" * 60)


if __name__ == "__main__":
    main()
