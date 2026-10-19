import json,pathlib,statistics
root=pathlib.Path(__file__).resolve().parent
labels=('native','flashinfer','vllm')
data={name:json.loads((root/f'{name}-results.json').read_text())['results'] for name in labels}
rows=[]
for i,n in enumerate((128,1024)):
 a,b,c=(data[name][i] for name in labels)
 def summary(row):
  return {**{k:v for k,v in row.items() if k!='raw'},'decode_min':min(x['decode_tokens_s'] for x in row['raw']),'decode_max':max(x['decode_tokens_s'] for x in row['raw'])}
 def parity(x,y):
  pairs=list(zip(x['raw'],y['raw']))
  matching=[(u,v) for u,v in pairs if u['token_ids']==v['token_ids']]
  differences=[{'request_index':i,'first_different_token':next(j for j,(a,b) in enumerate(zip(u['token_ids'],v['token_ids'])) if a!=b),'different_positions':sum(a!=b for a,b in zip(u['token_ids'],v['token_ids']))} for i,(u,v) in enumerate(pairs) if u['token_ids']!=v['token_ids']]
  gain=100*(statistics.median(v['decode_tokens_s'] for u,v in matching)/statistics.median(u['decode_tokens_s'] for u,v in matching)-1) if matching else None
  return {'exact_sequences':len(matching),'requests':len(pairs),'differences':differences,'matching_sequences_decode_gain_pct':gain}
 rows.append(dict(prompt_tokens=n,engines={name:summary(data[name][i]) for name in labels},flashinfer_vs_native=dict(decode_gain_pct=100*(b['decode_tokens_s_median']/a['decode_tokens_s_median']-1),ttft_reduction_pct=100*(1-b['ttft_median_ms']/a['ttft_median_ms']),decode_ms_saved=1000/a['decode_tokens_s_median']-1000/b['decode_tokens_s_median']),vllm_vs_flashinfer=dict(decode_faster_pct=100*(c['decode_tokens_s_median']/b['decode_tokens_s_median']-1),flashinfer_decode_below_vllm_pct=100*(1-b['decode_tokens_s_median']/c['decode_tokens_s_median'])),native_flashinfer_parity=parity(a,b),flashinfer_vllm_parity=parity(b,c)))
(root/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
