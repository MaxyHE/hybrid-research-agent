import json,time,hashlib,argparse,os
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['dev','test'],required=True);a=p.parse_args();root=a.root
MODEL=root/'models/Qwen3-Reranker-0.6B';out=root/(a.phase+'_scores.jsonl'); meta=root/(a.phase+'_runtime.json')
assert not out.exists(),'Refusing to overwrite scores'
tasks={'A':'Given a web search query, retrieve relevant passages that answer the query','B':'Given a scientific claim, retrieve research papers that provide evidence supporting or refuting the claim.'}
if a.phase=='test':
 sel=json.loads((root/'selection.json').read_text());assert sel['selected_arm'] in ('A','B'); tasks={sel['selected_arm']:tasks[sel['selected_arm']]}
rows=json.loads((root/(a.phase+'_candidates.json')).read_text())['rows'];corpus={str(r['_id']):r for r in map(json.loads,(root/'corpus.jsonl').read_text().splitlines())}
torch.set_num_threads(4);assert torch.cuda.is_available();torch.manual_seed(42)
tokenizer=AutoTokenizer.from_pretrained(MODEL,padding_side='left',local_files_only=True)
load=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();load_seconds=time.perf_counter()-load
no=tokenizer.convert_tokens_to_ids('no');yes=tokenizer.convert_tokens_to_ids('yes');assert no!=yes
prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
pre=tokenizer.encode(prefix,add_special_tokens=False);suf=tokenizer.encode(suffix,add_special_tokens=False);maximum=4096

def score_pairs(pairs):
 texts=[f'<Instruct>: {task}\n<Query>: {q}\n<Document>: {doc}' for task,q,doc in pairs]
 raw=tokenizer(texts,padding=False,truncation=False,add_special_tokens=False)['input_ids']
 truncated=[len(t)+len(pre)+len(suf)>maximum for t in raw]
 inputs=[pre+t[:maximum-len(pre)-len(suf)]+suf for t in raw]
 # Sort per query to reduce padding while preserving original candidate order in receipts.
 order=sorted(range(len(inputs)),key=lambda i:len(inputs[i]));values=[None]*len(inputs);elapsed=0
 for start in range(0,len(order),8):
  batch_ids=order[start:start+8];features=tokenizer.pad({'input_ids':[inputs[i] for i in batch_ids]},padding=True,return_tensors='pt');features={k:v.cuda() for k,v in features.items()}
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.inference_mode():logits=model(**features,use_cache=False,logits_to_keep=1).logits[:,-1,:].float();diff=(logits[:,yes]-logits[:,no]).cpu().tolist()
  torch.cuda.synchronize();elapsed+=time.perf_counter()-t
  for i,s in zip(batch_ids,diff):values[i]=float(s)
 return values,truncated,elapsed,[len(t) for t in inputs]
# A synthetic positive/negative control tests the yes/no scoring implementation, never used for model or prompt selection.
control,_,_,_=score_pairs([(tasks[next(iter(tasks))],'What is the capital of China?','The capital of China is Beijing.'),(tasks[next(iter(tasks))],'What is the capital of China?','Gravity attracts objects with mass.')]);assert control[0]>control[1]
print('MODEL_READY',transformers.__version__,torch.cuda.get_device_name(),control,flush=True)
started=time.perf_counter();total=trunc=0;arm_seconds={k:0. for k in tasks}; lengths=[]
with out.open('w') as receipt:
 for i,row in enumerate(rows,1):
  result={'query_id':row['query_id'],'arms':{}}
  for arm,task in tasks.items():
   docs=[f"# {corpus[d]['title']}\n\nSciFact corpus id: {d}\n\n{corpus[d]['text']}\n" for d in row['candidate_ids']]
   scores,ts,t,lens=score_pairs([(task,row['query'],doc) for doc in docs]);arm_seconds[arm]+=t;total+=len(scores);trunc+=sum(ts);lengths+=lens
   result['arms'][arm]={'scores':[{'document_id':d,'logit_yes_minus_no':s,'truncated':tr} for d,s,tr in zip(row['candidate_ids'],scores,ts)],'gpu_forward_seconds':t}
  receipt.write(json.dumps(result)+'\n');receipt.flush()
  if i%10==0 or i==1:print(a.phase,i,'/',len(rows),'pairs',total,'elapsed',round(time.perf_counter()-started,1),flush=True)
info={'status':'complete','phase':a.phase,'queries':len(rows),'pairs':total,'truncated_pairs':trunc,'arm_gpu_forward_seconds':arm_seconds,'wall_seconds':time.perf_counter()-started,'model_load_seconds':load_seconds,'max_input_tokens_observed':max(lengths),'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','batch_size':8,'max_length':maximum,'attention':'sdpa','model':'Qwen/Qwen3-Reranker-0.6B','synthetic_control_logit_diffs':control,'candidate_sha256':hashlib.sha256((root/(a.phase+'_candidates.json')).read_bytes()).hexdigest(),'scoring_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tasks':tasks}
meta.write_text(json.dumps(info,indent=2)+'\n');print('SCORING_COMPLETE',json.dumps(info),flush=True)
