import pytest
from policy.Cosmos.compiled_forward import CompiledForward


class CompilerFailure(Exception):
    pass


def test_compilation_failure_finishes_current_call_and_disables_future_retries():
    calls=[];events=[]
    def eager(x, *, delta):calls.append('eager');return x+delta
    def compiled(*args, **kwargs):calls.append('compiled');raise CompilerFailure('shape guard limit')
    wrapper=CompiledForward(eager,compiled,(CompilerFailure,),events.append)
    assert wrapper(3,delta=2)==5
    assert wrapper(7,delta=4)==11
    assert calls==['compiled','eager','eager']
    assert events==[dict(error_type='CompilerFailure',error='shape guard limit')]
    assert wrapper.failure==events[0]


def test_healthy_compiled_path_preserves_argument_and_result_identity():
    sentinel=object();events=[]
    def compiled(x, *, marker):assert marker is sentinel;return x
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected eager fallback')
    wrapper=CompiledForward(forbidden,compiled,(CompilerFailure,),events.append)
    payload={'sequence_length':4309}
    assert wrapper(payload,marker=sentinel) is payload
    assert wrapper.failure is None and not events


@pytest.mark.parametrize('error',[ValueError('bad input'),RuntimeError('CUDA failure')])
def test_model_errors_are_not_hidden_as_compiler_fallback(error):
    events=[]
    def compiled(*args):raise error
    wrapper=CompiledForward(lambda:None,compiled,(CompilerFailure,),events.append)
    with pytest.raises(type(error),match=str(error)):wrapper()
    assert not events and wrapper.failure is None
