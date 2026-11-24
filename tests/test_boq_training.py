"""CPU sampling/recipe regressions; no Trainer.fit, downloads or training jobs."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision.transforms import v2 as T

from reproduction.boq_protocol import OFFICIAL, BoQTrainingModule, check_recipe, input_transform
from reproduction.train_dino_gsv_mixed import (EpochBatches, MixedViews, collate_places,
                                               generated_count, load_config, resume_epoch, trial_summary)


@pytest.fixture
def views(tmp_path):
    root, cache = tmp_path/'gsv', tmp_path/'cache'
    root.mkdir(); cache.mkdir()
    paths, labels, keys, offsets, records = [], [], [], [0], []
    for label in range(24):
        city = 'A' if label < 12 else 'B'
        keys.append([city, label])
        # The final place has fewer than K views and is excluded as upstream does.
        for image in range(6 if label < 23 else 3):
            name = f'{label}_{image}.jpg'
            Image.new('RGB', (13, 17), (30+image, 40+label, 50)).save(root/name)
            paths.append(name.encode()); labels.append(label)
        offsets.append(len(paths))
        if label not in (0, 5, 12):
            night = root/f'{label}_generated.png'
            Image.new('RGB', (13, 17), (4, 5, label)).save(night)
            records.append(dict(label=label, city=city, place_id=label, night_path=str(night)))
    for name, array in [('paths', paths), ('labels', labels), ('offsets', offsets)]:
        np.save(cache/f'{name}.npy', np.asarray(array))
    (cache/'manifest.json').write_text(json.dumps({'root': str(root)}))
    (cache/'place_keys.json').write_text(json.dumps(keys))
    config = dict(OFFICIAL, base_cache=str(cache), gsv_root=str(root), batch_size=5, epochs=2)
    return MixedViews(config, records=records), config


@pytest.mark.parametrize('ratio,denominator', [('real', None), ('8:1', 9), ('4:1', 5)])
def test_ratios_keep_full_place_pool_four_distinct_views_and_tail(views, ratio, denominator):
    data, config = views
    batches = list(data.batches(config['seed'], 0, ratio))
    assert len(batches) == 5 and len(batches[-1]) == 3*4
    assert sum(len(b) for b in batches) == 23*4
    generated = sum(i < 0 for b in batches for i in b)
    assert generated == (0 if denominator is None else 23*4//denominator)
    places = []
    for batch in batches:
        for group in np.asarray(batch).reshape(-1, 4):
            assert len(set(group)) == 4
            labels = [data.generated_labels[-1-i] if i < 0 else data.labels[i] for i in group]
            assert len(set(labels)) == 1
            assert sum(i < 0 for i in group) <= 1
            places.append(labels[0])
    assert sorted(places) == list(range(23))  # Includes places with no generated view.
    assert list(data.batches(config['seed'], 0, ratio, skip=2)) == batches[2:]


def test_ratio_trials_use_identical_place_order_and_real_candidates(views):
    data, config = views
    real = list(data.batches(config['seed'], 0, 'real'))
    for ratio in ('8:1', '4:1'):
        mixed = list(data.batches(config['seed'], 0, ratio))
        for original, changed in zip(real, mixed):
            assert all(a == b for a, b in zip(original, changed) if b >= 0)
    plans = [trial_summary(data, config, r) for r in ('real', '8:1', '4:1')]
    assert len({p['total_steps'] for p in plans}) == 1
    assert len({p['places'] for p in plans}) == 1
    assert len({p['epoch_counts'][0]['images'] for p in plans}) == 1


def test_cumulative_rounding_does_not_change_the_official_batch():
    for ratio, denominator in [('8:1', 9), ('4:1', 5)]:
        counts = [generated_count(ratio, 512, i*512) for i in range(denominator)]
        assert sum(counts) == 512
        assert max(counts) - min(counts) <= 1
    for ratio in ('2:1', '1:1'):
        with pytest.raises(ValueError, match='four distinct'):
            generated_count(ratio, 512)


def test_insufficient_generated_capacity_is_explicit(views):
    data, config = views
    data.night.clear()
    with pytest.raises(ValueError, match='insufficient distinct'):
        data.batch_layout(config['seed'], 0, '4:1')
    assert len(list(data.batches(config['seed'], 0, 'real'))) == 5


def test_training_transform_matches_upstream_uint8_randaugment_order():
    official = T.Compose([T.Resize((224, 224), interpolation=3),
        T.RandAugment(num_ops=3, magnitude=15, interpolation=2),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    image = torch.randint(0, 256, (3, 31, 35), dtype=torch.uint8)
    torch.manual_seed(7); expected = official(image)
    torch.manual_seed(7); actual = input_transform(224, train=True)(image)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not any(isinstance(op, T.RandAugment) for op in input_transform(322).transforms)


def test_collation_preserves_place_groups_and_returns_cpu_scalar_count(views):
    data, config = views
    batch = next(data.batches(config['seed'], 0, '4:1'))
    images, labels, generated = collate_places([data[i] for i in batch], k=4)
    assert images.shape == (5, 4, 3, 224, 224)
    assert labels.shape == (5, 4) and (labels == labels[:, :1]).all()
    assert type(generated) is int and generated == sum(i < 0 for i in batch)


def test_recipe_rejects_legacy_strategy_and_keeps_independent_output():
    config = load_config('configs/boq_dinob_official_ratios.json')
    check_recipe(config)
    assert config['batch_size']*config['img_per_place'] == 512
    assert config['ratios'] == ['real', '8:1', '4:1']
    assert config['root'] == 'runs/boq_dinob_official_ratios_v1'
    for key, value in [('batch_size', 90), ('warmup_epochs', 1), ('lr', 1e-5)]:
        with pytest.raises(ValueError, match=key):
            check_recipe(dict(config, **{key: value}))
    with pytest.raises(ValueError, match='Historical nonofficial'):
        load_config('configs/boq_dinob_gsv_ratio_sweep.json')


def module_fixture(tmp_path):
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(3, 3)
    model.aggregator = torch.nn.Linear(3, 3)
    config = dict(OFFICIAL, output_dir=str(tmp_path), log_every=10)
    module = BoQTrainingModule(model, config, {'fixture': True}, [], SimpleNamespace(log=lambda *a, **k: None))
    return module


def test_optimizer_warmup_and_milestones_match_official_step_sequence(tmp_path):
    module = module_fixture(tmp_path)
    opts, schedulers = module.configure_optimizers()
    optimizer, scheduler = opts[0], schedulers[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [g['lr'] for g in optimizer.param_groups] == [1e-4, 1e-4]
    assert [g['weight_decay'] for g in optimizer.param_groups] == [1e-4, 1e-4]
    state = SimpleNamespace(current_epoch=0, num_training_batches=3, global_step=0)
    module._trainer = state
    actual = []
    for epoch in range(22):
        state.current_epoch = epoch
        for batch in range(3):
            module.optimizer_step(epoch, batch, optimizer, lambda: None)
            actual.append(optimizer.param_groups[0]['lr'])
            state.global_step += 1
        scheduler.step()
    assert actual[:3] == pytest.approx([1e-4/30, 2e-4/30, 3e-4/30])
    assert actual[29] == pytest.approx(1e-4)
    assert actual[30] == pytest.approx(1e-5)
    assert actual[60] == pytest.approx(1e-6)
    # Full optimizer/scheduler state can restore the next scheduled LR exactly.
    opt2, sch2 = module.configure_optimizers()
    opt2[0].load_state_dict(deepcopy(optimizer.state_dict()))
    sch2[0].load_state_dict(deepcopy(scheduler.state_dict()))
    assert sch2[0].state_dict() == scheduler.state_dict()
    assert opt2[0].param_groups[0]['lr'] == optimizer.param_groups[0]['lr']


def test_checkpoint_identity_cannot_resume_a_different_ratio(tmp_path):
    module = module_fixture(tmp_path)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)
    with pytest.raises(RuntimeError, match='identity changed'):
        module.on_load_checkpoint({'identity': {'different_ratio': True}})


def test_next_epoch_is_set_before_loader_prefetch_and_resume_restores_it(tmp_path, views):
    data, config = views
    sampler = EpochBatches(data, config['seed'], '8:1')
    module = module_fixture(tmp_path)
    module._trainer = SimpleNamespace(current_epoch=0,
        train_dataloader=SimpleNamespace(batch_sampler=sampler))
    epoch0 = list(sampler)
    module.on_train_epoch_end()
    assert sampler.epoch == 1
    epoch1 = list(sampler)
    assert epoch1 != epoch0
    path = tmp_path/'last.ckpt'
    identity = {'ratio': '8:1'}
    torch.save(dict(identity=identity, global_step=data.steps, epoch=0), path)
    restored = EpochBatches(data, config['seed'], '8:1')
    restored.epoch = resume_epoch(path, identity, data.steps)
    assert list(restored) == epoch1
    with pytest.raises(RuntimeError, match='identity changed'):
        resume_epoch(path, {'ratio': '4:1'}, data.steps)
    torch.save(dict(identity=identity, global_step=data.steps+1, epoch=1), path)
    with pytest.raises(RuntimeError, match='completed epoch'):
        resume_epoch(path, identity, data.steps)


def test_streaming_validation_matches_full_search_and_releases_gallery(tmp_path):
    import faiss
    rng = np.random.default_rng(3)
    features = rng.normal(size=(27, 3)).astype('float32')
    full = faiss.IndexFlatL2(3)
    full.add(features[:20])
    _, expected = full.search(features[20:], 10)
    module = module_fixture(tmp_path)
    module.model = torch.nn.Identity()
    module.benchmarks = [SimpleNamespace(name='fixture', references=list(range(20)),
        queries=list(range(7)), positives=[np.array([p[0]]) for p in expected])]
    module.log = lambda *a, **kw: None
    module.on_validation_epoch_start()
    for start in range(0, 27, 6):
        stop = min(start+6, 27)
        module.validation_step((torch.from_numpy(features[start:stop]), torch.arange(start, stop)), start//6)
    state = module.validation_outputs[0]
    assert state['index'] is None
    np.testing.assert_array_equal(np.concatenate(state['predictions']), expected)
    assert sum(p.nbytes for p in state['predictions']) == 7*10*8
    module.on_validation_epoch_end()
    assert module.validation_outputs == {}
    assert json.loads((tmp_path/'epoch_01.json').read_text())['fixture']['R@1'] == 100.
