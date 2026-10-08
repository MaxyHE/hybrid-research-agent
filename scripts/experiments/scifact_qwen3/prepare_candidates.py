"""Build the offline dense/BM25/RRF prototype from a supplied original FAISS index."""
import argparse, json, re, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import faiss
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','queries','index','chunk-map','embedding','selection','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=('dev','test'),required=True)
    a=p.parse_args()
    corpus={str(r['_id']):r for r in map(json.loads,a.corpus.read_text().splitlines())}
    queries={str(r['_id']):r['text'] for r in map(json.loads,a.queries.read_text().splitlines())}
    selection=json.loads(a.selection.read_text())
    if a.phase=='dev':
        selected=selection['selected_query_ids']
    else:
        selected=selection['test_query_ids']
    # Selection contains IDs only; gold labels are not used for candidate construction.
    model=SentenceTransformer(str(a.embedding),device='cpu',local_files_only=True)
    model.max_seq_length=256
    vecs=model.encode([queries[q] for q in selected],normalize_embeddings=True).astype('float32')
    index=faiss.read_index(str(a.index));assert index.ntotal==14940
    faiss.omp_set_num_threads(1)
    _,found=index.search(vecs,64)
    chunk_map=json.loads(a.chunk_map.read_text())
    ids=sorted(corpus);stem=PorterStemmer();cache={}
    def tokens(text):
        words=re.findall('[a-z0-9]+',text.lower())
        for word in words:
            if word not in cache: cache[word]=stem.stem(word)
        return [cache[word] for word in words]
    post=defaultdict(list);length=[]
    for i,d in enumerate(ids):
        ts=tokens(corpus[d]['title']+' '+corpus[d]['text']);length.append(len(ts))
        for word,tf in Counter(ts).items():post[word].append((i,tf))
    length=np.array(length,float);norm=1.2*(.25+.75*length/length.mean())
    post={word:(np.array([i for i,_ in vals]),np.array([tf for _,tf in vals],float),math.log(1+(len(ids)-len(vals)+.5)/(len(vals)+.5))) for word,vals in post.items()}
    rows=[]
    for q,hits in zip(selected,found):
        dense=list(dict.fromkeys(chunk_map[str(int(h))] for h in hits));scores=np.zeros(len(ids))
        for word in set(tokens(queries[q])):
            if word in post:
                inds,tf,idf=post[word];scores[inds]+=idf*tf*2.2/(tf+norm[inds])
        bm=[ids[int(i)] for i in np.lexsort((np.arange(len(ids)),-scores))[:64]]
        fused={}
        for ranking in (dense,bm):
            for rank,d in enumerate(ranking,1):fused[d]=fused.get(d,0)+1/(60+rank)
        pool=sorted(fused,key=lambda d:(-fused[d],d))[:64]
        rows.append({'query_id':q,'query':queries[q],'candidate_ids':pool,'dense_pool':dense,'bm25_pool':bm})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'phase':a.phase,'rows':rows},indent=2)+'\n')
    print(f'Built {len(rows)} frozen {a.phase} candidate pools.')

if __name__=='__main__':
    main()
