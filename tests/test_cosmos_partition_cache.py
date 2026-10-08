import typing
import pytest
if not hasattr(typing,'override'):
    from typing_extensions import override
    typing.override=override
pytest.importorskip('cosmos_framework.model.generator.pointflow_fk_attention')
import torch
from cosmos_framework.model.generator.pointflow_fk_attention import make_partition
from tools.try_cosmos_compile import PartitionCache
from tests.test_cosmos_observed_frame_batch import assert_equal


def test_reuse_layout_refreshes_depth_and_does_not_mutate_prior_result():
    cache=PartitionCache(make_partition)
    ids=torch.tensor([5,3]);first_depth=torch.tensor([1.,2.])
    first=cache([8],[2,6],ids,first_depth)
    second_depth=torch.tensor([3.,4.])
    second=cache([8],[2,6],ids,second_depth)
    assert_equal(first,make_partition([8],[2,6],ids,first_depth))
    assert_equal(second,make_partition([8],[2,6],ids,second_depth))
    assert cache.counts=={'miss':1,'hit':1}


def test_cfg_layout_and_index_order_use_separate_entries():
    cache=PartitionCache(make_partition)
    for split,ids in [([2,6],[5,3]),([1,7],[5,3]),([2,6],[3,5])]:
        ids=torch.tensor(ids);depth=torch.tensor([1.,2.])
        assert_equal(cache([8],split,ids,depth),make_partition([8],split,ids,depth))
    assert cache.counts=={'miss':3}


def test_invalid_depth_rejected_even_on_cache_hit():
    cache=PartitionCache(make_partition);ids=torch.tensor([3,5])
    cache([8],[2,6],ids,torch.ones(2))
    with pytest.raises(ValueError):cache([8],[2,6],ids,torch.tensor([float('nan'),2.]))
