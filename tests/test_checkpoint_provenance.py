"""CPU metadata checks for asymmetric teacher/student source consistency."""
import pytest

from reproduction.evaluate import TRANSFORM, validate_inference_checkpoint


@pytest.fixture
def source_pair():
    config = dict(student_name='efficientvit_b2.r224_in1k', descriptor_dim=12288,
                  proj_channels=384, num_queries=64, num_layers=2, image_size=322,
                  precision='float32')
    state = dict(config=dict(config), model={}, provenance=dict(
        teacher_sha256='teacher-a', image_size=322, descriptor_dim=12288,
        transform=TRANSFORM, precision='float32'))
    return state, config


def test_legacy_training_provenance_is_accepted_without_new_msls_schema(source_pair):
    state, config = source_pair
    validate_inference_checkpoint(state, config, 'teacher-a')


def test_different_same_architecture_gallery_is_rejected(source_pair):
    state, config = source_pair
    with pytest.raises(ValueError, match='teacher_sha256'):
        validate_inference_checkpoint(state, config, 'teacher-b')


@pytest.mark.parametrize('key', ['student_name', 'descriptor_dim', 'proj_channels',
                                 'num_queries', 'num_layers', 'image_size'])
def test_each_model_configuration_field_is_required(source_pair, key):
    state, config = source_pair
    del state['config'][key]
    with pytest.raises(ValueError, match=f'config mismatch: {key}'):
        validate_inference_checkpoint(state, config, 'teacher-a')


@pytest.mark.parametrize('key', ['teacher_sha256', 'image_size', 'descriptor_dim',
                                 'transform', 'precision'])
def test_each_training_feature_provenance_field_is_required(source_pair, key):
    state, config = source_pair
    del state['provenance'][key]
    with pytest.raises(ValueError, match=f'provenance mismatch: {key}'):
        validate_inference_checkpoint(state, config, 'teacher-a')


@pytest.mark.parametrize('key, value', [('image_size', 224), ('descriptor_dim', 8448),
                                      ('transform', 'different resize'), ('precision', 'float16')])
def test_changed_training_feature_protocol_is_rejected(source_pair, key, value):
    state, config = source_pair
    state['provenance'][key] = value
    with pytest.raises(ValueError, match=f'provenance mismatch: {key}'):
        validate_inference_checkpoint(state, config, 'teacher-a')


def test_non_fp32_evaluation_configuration_is_rejected(source_pair):
    state, config = source_pair
    config['precision'] = 'float16'
    with pytest.raises(ValueError, match='precision=float32'):
        validate_inference_checkpoint(state, config, 'teacher-a')
