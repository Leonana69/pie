import asyncio,json,time,statistics,pathlib
import aiohttp
from transformers import AutoTokenizer
MODEL='/workspace/models/Qwen3.8-Flash-Next-FP8'
OUT=pathlib.Path('/workspace/qwen-flashinfer-benchmark')
async def main():
 tok=AutoTokenizer.from_pretrained(MODEL)
 rows=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
  async with s.post('http://127.0.0.1:8000/v1/chat/completions',json=dict(model='qwen-fp8',messages=[dict(role='user',content='What is 17 multiplied by 23? Give only the numeric answer.')],max_tokens=32,temperature=0,chat_template_kwargs=dict(enable_thinking=False))) as r:
   if r.status!=200: raise RuntimeError(await r.text())
   smoke=await r.json(); (OUT/'vllm-smoke.json').write_text(json.dumps(smoke,indent=2))
   answer=smoke['choices'][0]['message']['content']; print('SMOKE',answer,flush=True)
   if '391' not in answer: raise RuntimeError('Arithmetic smoke failed')
  async def run(n,idx,output=256):
   text=('Benchmark sample %d. '%idx)+('The quick brown fox jumps over the lazy dog. '*(n//8+30))
   ids=tok.encode(text,add_special_tokens=False)[:n]
   body=dict(model='qwen-fp8',prompt=ids,max_tokens=output,temperature=0,ignore_eos=True,stream=True,stream_options={'include_usage':True},return_token_ids=True)
   start=time.perf_counter(); first=None; last=None; usage={}; chunks=[]; token_ids=[]
   async with s.post('http://127.0.0.1:8000/v1/completions',json=body) as r:
    if r.status!=200: raise RuntimeError(await r.text())
    async for line in r.content:
     if not line.startswith(b'data: '): continue
     data=line[6:].strip()
     if data==b'[DONE]': break
     d=json.loads(data)
     if d.get('usage'): usage=d['usage']
     for c in d.get('choices',[]):
      token_ids.extend(c.get('token_ids') or [])
      if c.get('text'):
       now=time.perf_counter(); first=first or now; last=now; chunks.append(c['text'])
   end=time.perf_counter(); count=usage.get('completion_tokens')
   if not first or count!=output or len(token_ids)!=output: raise RuntimeError(f'Invalid completion {usage}')
   return dict(prompt_tokens=n,output_tokens=count,ttft_ms=1000*(first-start),latency_s=end-start,decode_tokens_s=(count-1)/(last-first),end_to_end_tokens_s=count/(end-start),text=''.join(chunks),stream_chunks=len(chunks),token_ids=token_ids)
  await run(128,999,256)
  for n in (128,1024):
   for concurrency in (1,):
    await asyncio.gather(*(run(n,10000+i,256) for i in range(concurrency)))
    batch=[]; start=time.perf_counter()
    for repeat in range(10):
     batch.extend(await asyncio.gather(*(run(n,repeat*concurrency+i) for i in range(concurrency))))
    elapsed=time.perf_counter()-start
    vals=sorted(x['ttft_ms'] for x in batch)
    result=dict(prompt_tokens=n,output_tokens=256,concurrency=concurrency,requests=len(batch),ttft_median_ms=statistics.median(vals),ttft_p95_ms=vals[min(len(vals)-1,int(.95*len(vals)))],decode_tokens_s_median=statistics.median(x['decode_tokens_s'] for x in batch),aggregate_output_tokens_s=sum(x['output_tokens'] for x in batch)/elapsed,raw=batch)
    rows.append(result); print(json.dumps({k:v for k,v in result.items() if k!='raw'}),flush=True)
    (OUT/'vllm-results.json').write_text(json.dumps(dict(runtime='vLLM',gpu='NVIDIA B200',ngram_cpu_offload=True,prompt_type="synthetic raw completion",prefix_caching=False,results=rows),indent=2))
asyncio.run(main())
