from transformers import AutoTokenizer
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams
import json


PROMPT = f'''以下是一串關於程式的對話，嚴格按照以下要求翻譯成流暢的繁體中文:
* 翻譯成流暢的台灣用語的繁體中文。
* 程式碼部分保留英文。
* 無需附加解釋或說明。
* 每段對話間以 [SEP] 區隔。
'''

TOK = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B-Instruct')

LOG = logging.getLogger(__name__)


class LLMPredictor:

    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct",
                       tensor_parallel_size=8,
                       enforce_eager=True,
                       gpu_memory_utilization=0.9,
                       dtype='half')
        self.params = SamplingParams(temperature=0.8,
                                     max_tokens=4096,
                                     top_p=0.4,
                                     top_k=30,
                                     )

    def __call__(self, batch: dict[str, str]) -> dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["trans_prompt"],
                                    self.params,
                                    use_tqdm=True
                                    )

        batch_messages, batch_prompt = [], []
        for output, system_prompt in zip(outputs, batch['system_prompt']):
            prompt = output.prompt
            resps = output.outputs[0].text
            batch_prompt.append(prompt)
            messages =  []
            role = 'user'
            res = split_resp(resps)

            for r in res:
                messages.append({'role': role, 'content': r})
                role = 'user' if role == 'assistant' else 'assistant'

            if not is_valid_conv(messages):
                messages = []
            batch_messages.append(messages)

        return {
            'messages_tw': batch_messages,
            'trans_prompt': batch_prompt
        }


def format_prompt(
    x
):
    messages: list[dict[str, str]] = x['messages']
    messages = [
        f'{m["content"]}\n[SEP]\n' for m in messages if m['role'] != 'system'
    ]

    res = PROMPT + '\n'.join(messages)

    res = TOK.apply_chat_template(
        [{'content': res, 'role': 'user'}], tokenize=False, add_generation_prompt=True)
    # x['trans_prompt'] =
    # print(x)
    return {'trans_prompt': res + '以下是翻譯結果：\n', 'system_prompt': x['messages'][0]['content'] if x['messages'][0]['role'] == 'system' else ''}


def split_resp(resp: str):
    resp = resp.split(
        '<|start_header_id|>assistant<|end_header_id|>\n\n以下是翻譯結果：')[-1].strip()

    resp = [r.strip()
            for r in resp.split('[SEP]')]
    resp = [r for r in resp if r]
    return resp


def is_valid_conv(x: list[dict[str, str]]):
    for a, b in zip(x, x[1:]):
        if a['role'] == b['role']:
            return False
    if x[-1]['role'] != 'assistant':
        return False
    return True


def main(
    # data_kwargs: dict,
    data_path: str,
    output_path: str,
    batch_size: int = 4096,
):
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    # data = data[:32]
    predictor = LLMPredictor()
    result = []
    for i in tqdm(list(range(0, len(data), batch_size))):
        batch = data[i:i + batch_size]
        batch = [format_prompt(x) for x in batch]
        batch = {k: [x[k] for x in batch] for k in batch[0]}
        batch_result = predictor(batch)
        batch_result = [{k: x[i] for k, x in batch_result.items()}
                        for i in range(len(batch_result['messages_tw']))]
        result.extend(batch_result)
        with open(output_path, 'w') as f:
            for r in result:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(output_path, 'w') as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
