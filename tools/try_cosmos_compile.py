"""Isolated, opt-in experiments; never changes production defaults or weights."""
import argparse,copy,json,os,sys,time,traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

class PartitionCache:
    """Request-local layout reuse. Never cache feature/depth/noise values."""
    def __init__(self,original):
        self.original=original;self.entries={};self.counts=Counter()
    def __call__(self,sample_lens,split_lens,geometry_indexes,depth):
        import torch
        ids=tuple(geometry_indexes.detach().cpu().tolist())
        key=(tuple(sample_lens),tuple(split_lens),ids,str(geometry_indexes.device),geometry_indexes.dtype)
        if key not in self.entries:
            result=self.original(sample_lens,split_lens,geometry_indexes,depth)
            order=geometry_indexes.argsort()
            self.entries[key]=(result,order);self.counts['miss']+=1
            return result
        template,order=self.entries[key]
        if depth.shape!=(len(ids),) or not torch.isfinite(depth).all():
            raise ValueError('Expected one finite depth coordinate per geometry token')
        self.counts['hit']+=1
        return replace(template,depth=depth[order])

@contextmanager
def cached_partition():
    from cosmos_framework.model.generator import pointflow_fk_attention as attention
    original=attention.make_partition;cache=PartitionCache(original)
    attention.make_partition=cache
    try:yield cache
    finally:attention.make_partition=original;cache.entries.clear()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import numpy as np,torch,yaml
    from tools.profile_cosmos_pointfk import observation
    from policy.Cosmos.pointfk_policy import PointFKPolicy
    config=yaml.safe_load(a.config.read_text());config.update(output_dir=str(a.output.resolve()/'model'),prediction_dir=str(a.output.resolve()/'predictions'))
    policy=PointFKPolicy(config);policy.reset(102100000);obs=observation(ROOT/'outputs/cosmos_local/pointfk_3000/live_episode_exec32_same_seed')
    report=dict(scope='single fixed observation; existing first-frame preprocessing enabled; no Isaac/RPC; experiments not default',torch=torch.__version__)
    def save():
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    def call():
        torch.cuda.synchronize();start=time.perf_counter();actions=policy.get_action(copy.deepcopy(obs));torch.cuda.synchronize()
        assert actions.shape==(32,52) and np.isfinite(actions).all()
        return time.perf_counter()-start
    call();call();report['eager_seconds']=[call() for _ in range(5)];save()
    # First verify every cached partition field on real inputs, outside timing.
    from tests.test_cosmos_observed_frame_batch import assert_equal
    from cosmos_framework.model.generator import pointflow_fk_attention as attention
    original=attention.make_partition;cache=PartitionCache(original)
    def checked(*args,**kwargs):
        result=cache(*args,**kwargs);assert_equal(original(*args,**kwargs),result);return result
    attention.make_partition=checked
    try:call()
    finally:attention.make_partition=original
    report['partition_validation']=dict(exact_equal=True,counts=dict(cache.counts));cache.entries.clear();save()
    values=[]
    for _ in range(5):
        with cached_partition() as cache:
            seconds=call();values.append(dict(seconds=seconds,counts=dict(cache.counts)))
    report['partition_cache']=values;save()
    net=policy.backend.model.net
    original_forward=net.language_model.forward
    start=time.perf_counter()
    try:
        net.language_model.forward=torch.compile(original_forward,fullgraph=True,mode='reduce-overhead',dynamic=False)
        first=call();report['whole_transformer_compile']=dict(first_seconds=first,warm_seconds=[call() for _ in range(3)])
    except Exception as e:
        report['whole_transformer_compile']=dict(error_type=type(e).__name__,error=str(e),elapsed_sec=time.perf_counter()-start)
        (a.output/'whole_compile_error.txt').write_text(traceback.format_exc())
    finally:
        net.language_model.forward=original_forward
        torch._dynamo.reset();save()
    # Capture one representative generation MLP input to compare its fixed-input output.
    layers=list(net.language_model.model.layers);module=layers[0].mlp_moe_gen
    captured=[]
    def capture(mod,args):
        if not captured:captured.append(args[0].detach().clone())
    handle=module.register_forward_pre_hook(capture);call();handle.remove()
    replacements=[]
    try:
        with torch.inference_mode():
            x=captured[0];expected=module(x).clone()
            start=time.perf_counter()
            compiled=torch.compile(module.forward,fullgraph=True,mode='reduce-overhead',dynamic=False)
            actual=compiled(x).clone();torch.cuda.synchronize()
            diff=(actual.float()-expected.float());scale=expected.float().square().mean().sqrt()
            report['mlp_fixed_input']=dict(shape=list(x.shape),dtype=str(x.dtype),first_compile_sec=time.perf_counter()-start,max_abs_diff=float(diff.abs().max()),relative_rms=float(diff.square().mean().sqrt()/scale),finite=bool(actual.isfinite().all()))
        save()
        for layer in layers:
            m=layer.mlp_moe_gen;replacements.append((m,m.forward));m.forward=torch.compile(m.forward,fullgraph=True,mode='reduce-overhead',dynamic=False)
        first=call();call()
        report['compiled_generation_mlp']=dict(layers=len(layers),first_seconds=first,warm_seconds=[call() for _ in range(5)],peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except Exception as e:
        report['compiled_generation_mlp']=dict(error_type=type(e).__name__,error=str(e));(a.output/'mlp_compile_error.txt').write_text(traceback.format_exc())
    finally:
        for m,f in replacements:m.forward=f
        save()
    print('REPORT',json.dumps(report),flush=True)

if __name__=='__main__':main()
