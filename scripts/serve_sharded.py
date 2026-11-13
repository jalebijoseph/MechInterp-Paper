"""Load BF16 Qwen3-32B on two GPUs; reuse the original jpeers inference runtime."""
import argparse
import json
import logging
import os
from pathlib import Path


def main():
    import torch
    import transformers
    import uvicorn
    from jpeers import server
    from jpeers.registry import get_model_spec
    from jpeers.runtime import LensedModel

    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=18100)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    os.environ['JPEERS_MODEL']='qwen3-32b'
    provenance_path=root/'model-provenance.json'
    provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}

    def load_model():
        if torch.cuda.device_count()!=2:
            raise RuntimeError('The 32B job requires exactly two visible GPUs')
        revision=os.environ.get('JPEERS_MODEL_REVISION') or provenance['model_revision']
        model_path=os.environ.get('JPEERS_MODEL_PATH') or 'Qwen/Qwen3-32B'
        lens_path=os.environ.get('JPEERS_LENS_PATH') or provenance['lens_path']
        spec=get_model_spec('qwen3-32b',lens_path=lens_path)
        tok=transformers.AutoTokenizer.from_pretrained(model_path,revision=revision)
        hf=transformers.AutoModelForCausalLM.from_pretrained(
            model_path,revision=revision,dtype=torch.bfloat16,
            device_map='balanced',max_memory={0:'34GiB',1:'34GiB'})
        placements={str(v).replace('cuda:', '') for v in hf.hf_device_map.values()}
        if 'cpu' in placements or 'disk' in placements or not {'0','1'}.issubset(placements):
            raise RuntimeError(f'Expected GPU-only model sharding across two devices: {placements}')
        if hf.model.norm.weight.device != hf.lm_head.weight.device:
            raise RuntimeError('Final normalization and output head must share a GPU')
        (root/'device-map.json').write_text(json.dumps(hf.hf_device_map,indent=2))
        logging.info('Loaded pinned Qwen3-32B BF16 across two GPUs')
        return LensedModel(spec,hf_model=hf,tokenizer=tok,device='auto',dtype='bf16')

    server.load_model_from_env=load_model
    uvicorn.run(server.app,host='127.0.0.1',port=args.port,workers=1)

if __name__=='__main__':
    main()
