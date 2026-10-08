"""Compare eager/compiled Transformer on identical captured CFG inputs."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--dynamic',action='store_true');p.add_argument('--compile-mode',default='reduce-overhead');p.add_argument('--eager-repeat',action='store_true');p.add_argument('--preserve-precision',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch,numpy as np,yaml
    from tools.profile_cosmos_pointfk import observation
    from policy.Cosmos.pointfk_policy import PointFKPolicy
    cfg=yaml.safe_load((ROOT/'outputs/cosmos_local/pointfk_3000/live_episode_exec32_same_seed/policy.yaml').read_text());cfg.update(output_dir=str(a.output.resolve()/'model'),prediction_dir=str(a.output.resolve()/'predictions'))
    policy=PointFKPolicy(cfg);policy.reset(102100000);model=policy.backend.model.net.language_model
    eager=model.forward;inputs=[]
    def capture(*args,**kwargs):
        if len(inputs)<2:inputs.append(copy.deepcopy((args,kwargs)))
        return eager(*args,**kwargs)
    model.forward=capture
    try:policy.get_action(observation(ROOT/'outputs/cosmos_local/pointfk_3000/live_episode_exec32_same_seed'))
    finally:model.forward=eager
    if a.preserve_precision:
        torch._inductor.config.emulate_precision_casts=True
        torch._inductor.config.force_same_precision=True
    compiled=eager if a.eager_repeat else torch.compile(eager,fullgraph=True,mode=a.compile_mode,dynamic=a.dynamic)
    (a.output/'mode.json').write_text(json.dumps(dict(dynamic=a.dynamic,compile_mode=a.compile_mode,eager_repeat=a.eager_repeat,preserve_precision=a.preserve_precision,model_training=policy.backend.model.training,transformer_training=model.training),indent=2))
    results=[]
    def tensors(value,path='output'):
        if isinstance(value,torch.Tensor):return {path:value}
        if isinstance(value,dict):
            return {k:v for key,item in value.items() for k,v in tensors(item,path+'/'+str(key)).items()}
        if isinstance(value,(tuple,list)):
            return {k:v for i,item in enumerate(value) for k,v in tensors(item,path+'/'+str(i)).items()}
        return {}
    for i,(args,kwargs) in enumerate(inputs):
        with torch.inference_mode():
            torch.cuda.synchronize();t=time.perf_counter();old=copy.deepcopy(eager(*args,**kwargs));torch.cuda.synchronize();old_sec=time.perf_counter()-t
            torch.compiler.cudagraph_mark_step_begin();t=time.perf_counter();new=copy.deepcopy(compiled(*args,**kwargs));torch.cuda.synchronize();first_sec=time.perf_counter()-t
            differences={};old_values=tensors(old);new_values=tensors(new)
            assert old_values and old_values.keys()==new_values.keys()
            for key,x in old_values.items():
                y=new_values[key];assert x.shape==y.shape and x.dtype==y.dtype
                if x.is_floating_point():
                    error=x.float()-y.float();den=x.float().square().mean().sqrt().clamp_min(1e-12)
                    differences[key]=dict(shape=list(x.shape),exact_equal=bool(torch.equal(x,y)),max_abs=float(error.abs().max()),relative_rms=float(error.square().mean().sqrt()/den),finite=bool(y.isfinite().all()))
                else:differences[key]=dict(exact_equal=bool(torch.equal(x,y)))
            warm=[]
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin();torch.cuda.synchronize();t=time.perf_counter();compiled(*args,**kwargs);torch.cuda.synchronize();warm.append(time.perf_counter()-t)
            results.append(dict(cfg_branch=i,eager_seconds=old_sec,compiled_first_seconds=first_sec,compiled_warm_seconds=warm,differences=differences))
            (a.output/'comparison.json').write_text(json.dumps(results,indent=2));print('BRANCH',json.dumps(results[-1]),flush=True)
    if a.preserve_precision:
        # Time the actual policy, excluding the first call with these guards.
        model.forward=compiled
        whole=[]
        try:
            obs=observation(ROOT/'outputs/cosmos_local/pointfk_3000/live_episode_exec32_same_seed')
            for _ in range(4):
                torch.cuda.synchronize();t=time.perf_counter()
                result=policy.get_action(copy.deepcopy(obs));torch.cuda.synchronize()
                assert result.shape==(32,52) and np.isfinite(result).all()
                whole.append(time.perf_counter()-t)
            (a.output/'whole_policy_seconds.json').write_text(json.dumps(dict(first=whole[0],warm=whole[1:]),indent=2))
        finally:model.forward=eager
    from torch._dynamo.utils import counters
    (a.output/'compile_counters.json').write_text(json.dumps({k:dict(v) for k,v in counters.items()},indent=2))

if __name__=='__main__':main()
