import json
import logging
import os
from os import path as osp
import sys

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from basicsr.data import build_dataloader, build_dataset
from basicsr.losses.ocr_loss import load_ocr_loss
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.inference_benchmark import InferenceBenchmark
from basicsr.utils.options import dict2str, parse_options


def load_test_ocr(model, opt):
    ocr_opt = opt.get('ocr_opt') or (opt.get('train') or {}).get('ocr_opt')
    if not ocr_opt:
        raise ValueError('Configure ocr_opt (or train.ocr_opt) with load and alphabet to evaluate OCR.')
    input_size = ocr_opt.get('input_size')
    model.cri_ocr = load_ocr_loss(
        checkpoint_path=os.path.expanduser(ocr_opt['load']),
        alphabet=ocr_opt['alphabet'],
        device=model.device,
        loss_weight=ocr_opt.get('loss_weight', 0.01),
        input_size=tuple(input_size) if input_size is not None else None,
        mean=ocr_opt.get('mean'),
        std=ocr_opt.get('std'),
    )
    model.cri_ocr.eval()
    model.cri_ocr.requires_grad_(False)


def test_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    if opt.get('dist'):
        raise ValueError('Run this benchmark with --launcher none to avoid duplicated test results.')
    torch.backends.cudnn.benchmark = True

    val_opt = opt.setdefault('val', {})
    val_opt.setdefault('save_img', False)
    val_opt.setdefault('suffix', None)
    val_opt['ocr_metrics'] = val_opt.get('ocr_metrics') or {}
    val_opt['ocr_metrics']['enabled'] = True
    # Testing evaluates one LR per track, including with older YAMLs.
    val_opt['ocr_metrics']['temporal_fusion'] = {'enabled': False}
    benchmark_opt = opt.get('test') or {}
    # Exactly one forward per selected LR: no additional warmup forwards.
    benchmark_opt['warmup_iterations'] = 0
    opt['test'] = benchmark_opt
    warmup_iterations = 0
    dashboard_opt = benchmark_opt.get('ocr_dashboard') or {}
    dashboard_enabled = dashboard_opt.get('enabled', True)
    dashboards = []
    save_ocr_images = (benchmark_opt.get('save_ocr_images') or {}).get('enabled', True)

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info('Test mode: first LR by filename per track; one SR/OCR pass, no majority vote or warmup forwards.')

    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        if dataset_opt['phase'] == 'train':
            raise ValueError('Use datasets.test or datasets.val in the test YAML, not datasets.train.')
        if dataset_opt['type'] == 'LPSRPlateDataset':
            dataset_opt['single_lr_per_track'] = True
        test_set = build_dataset(dataset_opt)
        if len(test_set) == 0:
            raise ValueError(f"Empty test dataset: {dataset_opt['name']}")
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=False,
            sampler=None, seed=opt['manual_seed'])
        logger.info('Test dataset %s: %d items', dataset_opt['name'], len(test_set))
        test_loaders.append(test_loader)
    if not test_loaders:
        raise ValueError('No test datasets configured.')
    logger.info(dict2str(opt))

    model = build_model(opt)
    network = model.net_g_ema if hasattr(model, 'net_g_ema') else model.net_g
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    logger.info('SR model parameters: %d (%.3f M); OCR parameters excluded.',
                parameter_count, parameter_count / 1e6)
    load_test_ocr(model, opt)

    # Synchronize every device used by DataParallel, or the model's sole device.
    device_ids = getattr(network, 'device_ids', None) or [model.device]
    def synchronize():
        if model.device.type == 'cuda':
            for device in device_ids:
                torch.cuda.synchronize(device)

    summary = {
        'sr_parameters': parameter_count,
        'sr_checkpoint': opt['path'].get('pretrain_network_g'),
        'device': str(model.device),
        'torch_version': torch.__version__,
        'datasets': [],
    }
    if model.device.type == 'cuda':
        summary['gpu'] = torch.cuda.get_device_name(model.device)

    for test_loader in test_loaders:
        name = test_loader.dataset.opt['name']
        logger.info('Testing %s...', name)
        temporal = bool((val_opt['ocr_metrics'].get('temporal_fusion') or {}).get('enabled', False))
        timer = InferenceBenchmark(warmup_iterations, synchronize=synchronize)
        dashboard = None
        if dashboard_enabled:
            from basicsr.utils.ocr_dashboard import OCRDashboard
            ocr_opt = opt.get('ocr_opt') or (opt.get('train') or {}).get('ocr_opt')
            dashboard = OCRDashboard(
                name=name,
                output_dir=osp.join(opt['path']['results_root'], 'ocr_dashboard', str(len(dashboards) + 1)),
                alphabet=ocr_opt['alphabet'], parameter_count=parameter_count, benchmark=timer,
                temporal=temporal, update_interval=dashboard_opt.get('update_interval', 0.5),
                top_k=dashboard_opt.get('top_k', 10),
            )
            dashboards.append(dashboard)
            model.ocr_dashboard = dashboard
        model.inference_benchmark = timer
        image_store = None
        if save_ocr_images:
            from basicsr.utils.ocr_result_store import OCRResultStore
            image_store = OCRResultStore(
                osp.join(opt['path']['results_root'], 'ocr_images', str(len(summary['datasets']) + 1)), name)
            model.ocr_result_store = image_store
        try:
            # Numeric iteration also supports the OCR example_freq option.
            result = model.validation(test_loader, current_iter=0, tb_logger=None,
                                      save_img=val_opt['save_img'])
        finally:
            del model.inference_benchmark
            if dashboard is not None:
                del model.ocr_dashboard
            if image_store is not None:
                image_store.close()
                del model.ocr_result_store
        timing = timer.summary()
        if timing['inference_count'] == 0:
            raise RuntimeError(f'No inference was measured in {name}. Check its LR files.')
        row = dict(result)
        row['ocr_scope'] = 'single_lr_per_track'
        row['lr_selection'] = ('first_by_filename' if test_loader.dataset.opt['type'] == 'LPSRPlateDataset'
                               else 'dataset_default')
        row['timing'] = timing
        if image_store is not None:
            row['ocr_image_review'] = image_store.summary()
            logger.info('SR review images: %d; by correct characters: %s; saved to %s',
                        image_store.count, dict(sorted(image_store.counts.items())), image_store.root)
        if dashboard is not None:
            row['ocr_dashboard_files'] = dashboard.finish()
        summary['datasets'].append(row)
        logger.info(
            'Test summary %s | SR parameters: %d | SR inference: %.3f +/- %.3f ms '
            '(mean +/- population std, %d calls, %d images, warmup excluded) | '
            'OCR plate accuracy [%s]: %.2f%%',
            name, parameter_count, timing['mean_ms'], timing['std_ms'],
            timing['inference_count'], timing['image_count'], row['ocr_scope'], 100 * result['ocr'])

    summary_path = osp.join(opt['path']['results_root'], 'test_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    logger.info('Test summary saved to %s', summary_path)
    if dashboards and dashboard_opt.get('block_on_finish', True):
        OCRDashboard.keep_open(dashboards)
    return summary


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
