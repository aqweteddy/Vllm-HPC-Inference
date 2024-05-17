from transformers import AutoTokenizer
import torch
import random
import time
import logging
from tqdm import tqdm
from vllm import EngineArgs, LLM, RequestOutput, SamplingParams
import json


PROMPT = '''針對我提供的 USER 與 ASSISTANT 的多輪對話進行評估，評估的三項標準如下：

0. 指令準確度：評估 USER 的指令是否清晰明確，並且是否能夠幫助 ASSISTANT 了解 USER 的需求。
    - 1分：指令不清晰，ASSISTANT 無法了解 USER 的需求。
    - 2分：指令不夠清晰，ASSISTANT 能夠稍微了解 USER 的需求，但無法明確回覆。
    - 3分：指令基本清晰，USER 表達基本需求，但有改進空間。
    - 4分：指令很清晰，ASSISTANT 能夠根據 USER 的需求進行回答。
    - 5分：指令非常清晰，ASSISTANT 完全了解 USER 的需求，且能夠完全滿足。
1. 指令遵循度與有幫助性：評估 ASSISTANT 的回答是否根據 USER 的指令進行，並且是否滿足 USER 需求。
    - 1分：回答完全未按照指令進行，也未能滿足 USER 需求。
    - 2分：回答試圖遵循指令但大部分未成功，對 USER 需求的滿足程度很低。
    - 3分：回答部分遵循指令，對 USER 需求有一定程度的滿足。
    - 4分：回答較好地遵循了指令，並且在很大程度上滿足了 USER 需求。
    - 5分：回答完全按照指令進行，並且完全滿足 USER 需求。
2. 內容正確性：評估模型提供訊息的正確性、專業性，以及是否無誤導、虛假資訊或含有中國相關內容。
    - 1分：提供的訊息大多不正確或大量幻想，含有誤導或虛假資訊，或是含有中國相關內容。
    - 2分：提供的訊息有部分不正確或幻想，可能含有少量誤導性或虛假資訊或是含有中國相關內容。
    - 3分：提供的訊息基本正確和專業，但有改進空間。
    - 4分：提供的訊息都是正確，不含誤導或虛假資訊，但專業度略有不足。
    - 5分：提供的訊息完全正確和專業，完全無誤導或虛假資訊。
3. 對話一致性：確保在對話中保持訊息和主題的一致性，包括各回合間的訊息一致性。
    - 1分：對話中的訊息和主題大部分時間不一致。
    - 2分：對話中的訊息和主題有時不一致。
    - 3分：對話中的訊息和主題基本一致。
    - 4分：對話中的訊息和主題大部分時間一致。
    - 5分：對話中的訊息和主題完全一致。
4. 綜合性評分: 綜合以上三項標準，給出綜合性評分。
在評估的過程中，牢記這些評估標準。先給出評分的理由，最後給出以下格式:

理由: 分析每個項目的缺點。
指令遵循度與有幫助性: 0-5的整數
內容正確性: 0-5的整數
多輪對話一致性: 0-5的整數
綜合性評分: 0-5的整數

注意：在給出評估時，僅按照上述格式提供輸出。

[多輪對話開始]
{}
[多輪對話結束]
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
                                     max_tokens=8192,
                                     top_p=0.4,
                                     top_k=30,
                                     )

    def __call__(self, batch: dict[str, str]) -> dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["judge_prompt"],
                                    self.params,
                                    use_tqdm=True
                                    )
        
        batch_judge, batch_score, batch_prompt = [], [], []
        for output in outputs:
            prompt = output.prompt
            resp = output.outputs[0].text
            batch_prompt.append(prompt)
            judge, score = get_score(resp)
            batch_score.append(score)
            batch_judge.append(judge)

        return {
            'score': batch_score,
            'judge': batch_judge,
            'judge_prompt': batch_prompt,
            'messages': batch['messages']
        }


def format_prompt(
    x
):
    messages: list[dict[str, str]] = x['messages']
    messages = [
        f'{m["role"].upper()}: {m["content"]}' for m in messages if m['role'] != 'system'
    ]

    res = PROMPT.format('\n'.join(messages))

    res = TOK.apply_chat_template(
        [{'content': res, 'role': 'user'}], tokenize=False, add_generation_prompt=True)
    return {'judge_prompt': res + '以下是評分結果：\n', 'messages': x['messages']}

def get_score(resp):
    resp = resp.split(
        '<|start_header_id|>assistant<|end_header_id|>\n\以下是評分結果：')[-1].strip()
    overall = ''
    for sent in resp.split('\n'):
        if '綜合性評分' in sent.lower():
            overall = sent
            break
        
    for spl in [' ', ':', '：']:
        try:    
            score = overall.split(spl)[-1].strip()
            score = score[0]
            return resp, int(score)
        except:
            pass
    return resp, -1

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
                        for i in range(len(batch_result['score']))]
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
