import torch
import cv2
import glob
import sys
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.ocr_loss import load_ocr_loss
from basicsr.metrics import calculate_metric
from basicsr.metrics.ocr_recognition import OCRRecognitionMetrics
from basicsr.utils import get_root_logger, imwrite, tensor2img, img2tensor
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.training_viewer import TrainingViewer
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

        self.training_viewer = None

        viewer_opt = self.opt.get(
            "train",
            {}
        ).get(
            "viewer",
            {}
        )

        if (
            self.is_train
            and viewer_opt.get("enabled", False)
        ):
            self.training_viewer = TrainingViewer()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.use_amp = train_opt.get("use_amp", False)

        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('contextual_opt'):
            self.cri_contextual = build_loss(train_opt['contextual_opt'])
        else:
            self.cri_contextual = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get("ssim_opt") is not None:
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None


        # Loss OCR
        ocr_opt = train_opt.get("ocr_opt")

        if ocr_opt is not None:
            input_size = ocr_opt.get("input_size")

            if input_size is not None:
                input_size = tuple(input_size)

            self.cri_ocr = load_ocr_loss(
                checkpoint_path=ocr_opt["load"],
                alphabet=ocr_opt["alphabet"],
                device=self.device,
                loss_weight=ocr_opt.get(
                    "loss_weight",
                    0.01,
                ),
                input_size=input_size,
                mean=ocr_opt.get("mean"),
                std=ocr_opt.get("std"),
            )

            self.cri_ocr.eval()
            self.cri_ocr.requires_grad_(False)

            logger = get_root_logger()
            logger.info(
                f"OCR loss carregada de: {ocr_opt['load']}"
            )
        else:
            self.cri_ocr = None


        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if data.get('gt') is not None:
            self.gt = data['gt'].to(self.device)
        elif hasattr(self, 'gt'):
            del self.gt

        self.lq_path = data["lq_path"]
        self.gt_path = data.get("gt_path")

        self.label = data.get("label")

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad(set_to_none=True)
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style, l_context = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
            if l_context is not None:
                l_total += l_context
                loss_dict['l_context'] = l_context
        # contextual loss
        if self.cri_contextual:

            l_context = self.cri_contextual(self.output, self.gt)

            l_total += l_context

            loss_dict['l_context'] = l_context

        # OCR loss
        if self.cri_ocr:
            ''' Old version, without cross entropy
            l_ocr = self.cri_ocr(
                self.output,
                self.gt,
                self.label,
            )
            '''
            self.cri_ocr.eval()
            for parameter in self.cri_ocr.parameters():
                parameter.requires_grad_(False)


            l_ocr = self.cri_ocr.cross_entropy_loss(self.output, self.label,)

            if not l_ocr.requires_grad:
                raise RuntimeError("OCR loss is detached from the computational graph.")

            l_total += l_ocr
            loss_dict["l_ocr"] = l_ocr

        # SSIM loss
        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            l_total += l_ssim
            loss_dict["l_ssim"] = l_ssim


        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()


    def _run_test_inference(self):
        benchmark = getattr(self, 'inference_benchmark', None)
        if benchmark is None:
            self.test()
        else:
            benchmark.run(self.test, tuple(self.lq.shape))

    def _update_ocr_dashboard(self, results, processed, total):
        dashboard = getattr(self, 'ocr_dashboard', None)
        if dashboard is not None:
            dashboard.update(results, processed, total)

    def _save_ocr_result(self, target, prediction, lq_path, gt_path=None,
                         hr_candidates=None, image_index=0):
        store = getattr(self, 'ocr_result_store', None)
        if store is not None:
            store.save(
                sr_rgb=tensor2img(self.output[image_index], rgb2bgr=False),
                lr_rgb=tensor2img(self.lq[image_index], rgb2bgr=False),
                target=target, prediction=prediction, lq_path=lq_path,
                gt_path=gt_path, hr_candidates=hr_candidates,
            )

    @staticmethod
    def _update_validation_progress(current, total, bar_width=30):
        if total <= 0:
            return

        progress = current / total
        filled = int(bar_width * progress)

        bar = "=" * filled + ">" + "." * max(0, bar_width - filled - 1)

        message = (
            f"\rValidation: [{bar}] "
            f"{current}/{total} "
            f"({progress * 100:6.2f}%)"
        )

        sys.stdout.write(message)

        # Só cria uma nova linha quando a validação terminar.
        if current >= total:
            sys.stdout.write("\n")

        sys.stdout.flush()




    def nondist_validation_temporal(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        total = len(dataloader)

        dataset_name = dataloader.dataset.opt['name']
        val_opt = self.opt['val']
        ocr_metrics_opt = val_opt.get('ocr_metrics') or {}

        # this function always does ocr recognition
        if self.cri_ocr is None:
            raise RuntimeError(
                'val.ocr.metrics.enabled=true requires train.ocr_opt to load the OCR model.'
            )
        ocr_results = OCRRecognitionMetrics()
        ocr_examples = []
        example_freq = int(ocr_metrics_opt.get('example_freq', 0) or 0)
        log_examples = example_freq > 0 and current_iter % example_freq == 0
        max_examples = int(ocr_metrics_opt.get('num_examples', 5))

        
        with_metrics = val_opt.get('metrics') is not None
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        reference_metric_count = 0

        dataset = dataloader.dataset

        for idx, val_data in enumerate(dataloader):
            self._update_validation_progress(idx + 1, total)
            original_lq_path = val_data["lq_path"][0]
            folder = osp.dirname(original_lq_path)

            sample = dataset.samples[idx]
            gt_paths = val_data.get("gt_path")
            gt_path = gt_paths[0] if gt_paths else None
            label = val_data["label"]
            if isinstance(label, (list, tuple)):
                label = str(label[0])
            else:
                label = str(label)

            # Use the dataset index, including LR_ names and uppercase extensions.
            lq_paths = sorted(sample['lq_candidates'])

            if len(lq_paths) == 0:
                logger.warning(f"Nenhuma imagem LR encontrada em {folder}")
                continue

            # OCR prediction for all SR images from the track
            track_predictions = []

            img_gt = None

            for lq_path in lq_paths:
                img_lq, img_gt = dataset.load_sample(sample, lq_path, gt_path)

                # [C, H, W] -> [B, C, H, W]
                img_lq = img_lq.unsqueeze(0)

                current_data = {
                        "lq": img_lq,
                        "label": label,
                        "lq_path": lq_path,
                    }
                if img_gt is not None:
                    current_data.update(gt=img_gt.unsqueeze(0), gt_path=gt_path)

                # SR
                self.feed_data(current_data)
                self._run_test_inference()

                # OCR
                prediction = self.cri_ocr.predict(self.output)

                if isinstance(prediction, (list, tuple)):
                    if len(prediction) != 1:
                        raise ValueError(
                            f"Expected one OCR prediction, got {prediction}"
                        )
                    prediction = prediction[0]

                prediction = str(prediction).strip().upper()
                track_predictions.append(prediction)
                self._save_ocr_result(label, prediction, lq_path, gt_path,
                                      hr_candidates=sample.get('gt_candidates'))


            final_prediction = ocr_results.majority_vote(track_predictions)

            ocr_results.update([label], [final_prediction])
            self._update_ocr_dashboard(ocr_results, idx + 1, total)
            if log_examples and len(ocr_examples) < max_examples:
                ocr_examples.append((osp.basename(lq_path), label, prediction))


            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data = {'img': sr_img}
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if getattr(self, 'inference_benchmark', None) is None:
                torch.cuda.empty_cache()


            if save_img:
                img_name = osp.splitext(osp.basename(lq_path))[0]
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics and 'img2' in metric_data:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                reference_metric_count += 1


        if with_metrics:
            if reference_metric_count:
                for metric in self.metric_results:
                    self.metric_results[metric] /= reference_metric_count
                    self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            else:
                self.metric_results = {name: None for name in self.metric_results}
            get_root_logger().info(
                'Reference image metrics: %d/%d evaluated items have HR; items without HR are excluded.',
                reference_metric_count, total)

        self._log_ocr_metric_values(current_iter, dataset_name, ocr_results, ocr_examples, tb_logger, top_k=int(ocr_metrics_opt.get('top_k_errors', 10)),)

        return {
            'dataset': dataset_name,
            'ocr': ocr_results.plate_accuracy,
            'ocr_chars_any_position': ocr_results.character_accuracy_any_position,
            'ocr_details': ocr_results.to_dict(),
            'psnr': self.metric_results.get('psnr') if with_metrics else None,
            'reference_metric_count': reference_metric_count,
        }




    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        total = len(dataloader)
        dataset_name = dataloader.dataset.opt['name']
        val_opt = self.opt['val']
        with_metrics = val_opt.get('metrics') is not None

        ocr_metrics_opt = val_opt.get('ocr_metrics') or {}
        use_ocr_metrics = bool(ocr_metrics_opt.get('enabled', False))
        if use_ocr_metrics and self.cri_ocr is None:
            raise RuntimeError(
                'val.ocr_metrics.enabled=true requires train.ocr_opt to load the OCR model.'
            )
        ocr_results = OCRRecognitionMetrics() if use_ocr_metrics else None
        ocr_examples = []
        example_freq = int(ocr_metrics_opt.get('example_freq', 0) or 0)
        log_examples = use_ocr_metrics and example_freq > 0 and current_iter % example_freq == 0
        max_examples = int(ocr_metrics_opt.get('num_examples', 5))

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        reference_metric_count = 0

        for idx, val_data in enumerate(dataloader):
            self._update_validation_progress(idx + 1, total)
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self._run_test_inference()

            if use_ocr_metrics:
                labels = [str(label) for label in val_data['label']]
                predictions = self.cri_ocr.predict(self.output)
                for image_index, (label, prediction, lq_path) in enumerate(
                        zip(labels, predictions, val_data['lq_path'])):
                    gt_paths = val_data.get('gt_path')
                    self._save_ocr_result(
                        label, prediction, lq_path,
                        gt_paths[image_index] if gt_paths else None,
                        image_index=image_index,
                    )
                ocr_results.update(labels, predictions)
                self._update_ocr_dashboard(ocr_results, idx + 1, total)

                if log_examples and len(ocr_examples) < max_examples:
                    for label, prediction, path in zip(labels, predictions, val_data['lq_path']):
                        if len(ocr_examples) >= max_examples:
                            break
                        ocr_examples.append((osp.basename(path), label, prediction))

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data = {'img': sr_img}
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if getattr(self, 'inference_benchmark', None) is None:
                torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics and 'img2' in metric_data:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                reference_metric_count += 1

        if with_metrics:
            if reference_metric_count:
                for metric in self.metric_results:
                    self.metric_results[metric] /= reference_metric_count
                    self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
            else:
                self.metric_results = {name: None for name in self.metric_results}
            get_root_logger().info(
                'Reference image metrics: %d/%d evaluated items have HR; items without HR are excluded.',
                reference_metric_count, total)

        if use_ocr_metrics:
            self._log_ocr_metric_values(
                current_iter,
                dataset_name,
                ocr_results,
                ocr_examples,
                tb_logger,
                top_k=int(ocr_metrics_opt.get('top_k_errors', 10)),
            )

        return {
            'dataset': dataset_name,
            'ocr': ocr_results.plate_accuracy if use_ocr_metrics else None,
            'ocr_chars_any_position': (
                ocr_results.character_accuracy_any_position if use_ocr_metrics else None
            ),
            'ocr_details': ocr_results.to_dict() if use_ocr_metrics else None,
            'psnr': self.metric_results.get('psnr') if with_metrics else None,
            'reference_metric_count': reference_metric_count,
        }

    def _log_ocr_metric_values(
        self,
        current_iter,
        dataset_name,
        results,
        examples,
        tb_logger,
        top_k=10,
    ):
        logger = get_root_logger()
        lines = [
            f'OCR validation {dataset_name} @ {current_iter} iter',
            (f'\t# plate_accuracy: {results.plate_accuracy:.4f} '
             f'({results.correct_plates}/{results.total_plates})'),
            (f'\t# character_accuracy_any_position: '
             f'{results.character_accuracy_any_position:.4f} '
             f'({results.correct_characters_any_position}/{results.total_characters})'),
            '\t# accuracy_by_position:',
        ]

        for position, accuracy in results.position_accuracies().items():
            correct = results.position_correct[position]
            total = results.position_total[position]
            lines.append(
                f'\t  position_{position + 1}: {accuracy:.4f} ({correct}/{total})'
            )

        lines.append('\t# character_precision (correct / predicted):')
        for row in results.to_dict()['characters']:
            if row['target_total'] or row['predicted_total']:
                precision = f"{row['precision']:.4f}" if row['predicted_total'] else 'n/a'
                lines.append(f"\t  {row['character']}: {precision} "
                             f"({row['correct']}/{row['predicted_total']})")

        lines.append(f'\t# most_error_prone_characters (top {top_k}):')
        error_rows = results.most_error_prone_chars(top_k)
        if error_rows:
            for char, errors, total, error_rate in error_rows:
                lines.append(
                    f'\t  {char}: {errors}/{total} errors ({error_rate:.2%})'
                )
        else:
            lines.append('\t  none')

        lines.append(f'\t# most_common_confusions (top {top_k}):')
        confusion_rows = results.most_common_confusions(top_k)
        if confusion_rows:
            for expected, predicted, count in confusion_rows:
                lines.append(f'\t  {expected} -> {predicted}: {count}')
        else:
            lines.append('\t  none')

        if examples:
            lines.append('\t# examples:')
            for filename, target, prediction in examples:
                status = 'OK' if target == prediction else 'ERROR'
                lines.append(
                    f'\t  [{status}] {filename}: GT={target} | OCR={prediction}'
                )

        logger.info('\n'.join(lines))

        if tb_logger:
            prefix = f'ocr/{dataset_name}'
            tb_logger.add_scalar(
                f'{prefix}/plate_accuracy', results.plate_accuracy, current_iter
            )
            tb_logger.add_scalar(
                f'{prefix}/character_accuracy_any_position',
                results.character_accuracy_any_position,
                current_iter,
            )
            for char, precision in results.character_precisions().items():
                tb_logger.add_scalar(f'{prefix}/character_precision/{char}', precision, current_iter)
            for position, accuracy in results.position_accuracies().items():
                tb_logger.add_scalar(
                    f'{prefix}/position_{position + 1}_accuracy',
                    accuracy,
                    current_iter,
                )

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def save_best(self):
        """Save only the generator weights using the stable best.pth name."""
        if hasattr(self, 'net_g_ema'):
            self.save_network(
                [self.net_g, self.net_g_ema],
                'net_g',
                'best',
                param_key=['params', 'params_ema'],
                save_filename='best.pth',
            )
        else:
            self.save_network(
                self.net_g,
                'net_g',
                'best',
                save_filename='best.pth',
            )


    @torch.no_grad()
    def update_training_viewer(self, current_iter: int,) -> None:

        if self.training_viewer is None:
            return

        if not self.training_viewer.next_requested():
            return

        sample_idx = 0

        lq = self.lq[sample_idx:sample_idx + 1]
        hr_model = self.gt[sample_idx:sample_idx + 1]
        sr = self.output[sample_idx:sample_idx + 1]
        path = self.gt_path[sample_idx:sample_idx + 1]
        label = self.label[sample_idx]
        hr_original, hr_original_tensor = self._load_original_hr(sample_idx)
        hr_original_tensor = hr_original_tensor.to(self.device)

        ocr_hr_original = self.cri_ocr.predict(hr_original_tensor)
        ocr_hr_model = self.cri_ocr.predict(hr_model)
        ocr_sr = self.cri_ocr.predict(sr)

        # =========================================================
        # Caso predict retorne lista
        # =========================================================

        if isinstance(
            ocr_hr_original,
            (list, tuple),
        ):
            ocr_hr_original = ocr_hr_original[0]

        if isinstance(
            ocr_hr_model,
            (list, tuple),
        ):
            ocr_hr_model = ocr_hr_model[0]

        if isinstance(
            ocr_sr,
            (list, tuple),
        ):
            ocr_sr = ocr_sr[0]


        # =========================================================
        # Envia para processo da interface
        # =========================================================

        data = {
            "iteration": current_iter,

            "label": str(label),

            "lq": lq,

            "hr_original": hr_original,

            "hr_model": hr_model,

            "sr": sr,

            "ocr_hr_original": str(
                ocr_hr_original
            ),

            "ocr_hr_model": str(
                ocr_hr_model
            ),

            "ocr_sr": str(
                ocr_sr
            ),

            "path": str(
                path
            ),
        }

        self.training_viewer.send(data)

    def _load_original_hr(
        self,
        sample_idx: int,
    ):

        gt_path = self.gt_path[sample_idx]

        # OpenCV carrega em BGR
        hr_original = cv2.imread(
            gt_path,
            cv2.IMREAD_COLOR,
        )

        if hr_original is None:
            raise RuntimeError(
                f"Não foi possível carregar HR original: {gt_path}"
            )

        # ---------------------------------------------------------
        # Tensor para OCR
        #
        # Precisa reproduzir o mesmo pipeline do Dataset:
        # BGR -> RGB
        # uint8 [0,255] -> float32 [0,1]
        # HWC -> CHW
        # adiciona batch
        # ---------------------------------------------------------

        hr_rgb = cv2.cvtColor(
            hr_original,
            cv2.COLOR_BGR2RGB,
        )

        hr_original_tensor = (
            torch.from_numpy(
                hr_rgb.copy()
            )
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
        )

        return (
            hr_original,
            hr_original_tensor,
        )
