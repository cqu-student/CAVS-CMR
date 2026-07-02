from easydict import EasyDict as edict
import yaml
import pdb

"""
default config
"""
def modify_command_options(opts):
    if opts.method is not None:
        if opts.method == 'FT':
            pass
        if opts.method == 'LWF':
            opts.loss_kd = 100
        if opts.method == 'LWF-MC':
            opts.icarl = True
            opts.icarl_importance = 1
        if opts.method == 'ILT':
            opts.loss_kd = 100
            opts.loss_de = 100
        if opts.method == 'EWC':
            opts.regularizer = "ewc"
            opts.reg_importance = 500
        if opts.method == 'RW':
            opts.regularizer = "rw"
            opts.reg_importance = 100
        if opts.method == 'PI':
            opts.regularizer = "pi"
            opts.reg_importance = 500
        if opts.method == 'MiB':
            opts.loss_kd = 10
            opts.unce = True
            opts.unkd = True
            opts.init_balanced = True
        if opts.method == "PLOP":
            opts.pod = "local"
            opts.pod_factor = 0.01
            opts.pod_large_logits = True
            opts.pod_options = {"switch": {"after": {"extra_channels": "sum", "factor": 0.0005, "type": "local"}}}
            opts.pseudo = "entropy"
            # opts.ce_on_pseudo = True
            opts.threshold = 0.001
            opts.classif_adaptive_factor = True
            opts.init_balanced = True
        if opts.method == "SSUL":
            opts.unknown = False
            opts.pseudo_ssul = True
            opts.freeze = True
            opts.w_transfer = False

cfg = edict()
cfg.BATCH_SIZE = 2 # default 4
cfg.LAMBDA_1 = 0.5 # default: 0.5
cfg.MASK_NUM = 10 # 10 for fully supervised
cfg.NUM_CLASSES = 71 # 70 + 1 background
##############################class increment
cfg.dataset_name = "v2"  # v2s ,v2m, 这个可以通过meta数据修改读取的数据
cfg.step = 2
cfg.task_name = "60-5a"
cfg.step_checkpoint = "../../avs_scripts/avss/checkpoints/v2/pvt/60-5a/2/miou_PLOP_memory_abs_best.pth"
cfg.step_checkpoint_v = None
cfg.regularizer = None


cfg.method = "PLOP"
cfg.icarl_disjoint = False
cfg.icarl_bkg = False


cfg.loss_kd = 0
cfg.icarl = False
cfg.icarl_importance = 1
cfg.loss_de = 0
cfg.reg_importance = 1
cfg.unce = False
cfg.unkd = False
cfg.init_balanced = False
cfg.alpha = 1.


cfg.reg_alpha = 0.9
cfg.reg_no_normalize = False
cfg.reg_iterations = 10

cfg.pseudo = None
cfg.pod = None
cfg.old_classes = 0
cfg.pod_factor = 5.
cfg.pod_logits = False
cfg.pod_options = None
cfg.pesudo = None
cfg.threshold = 0.9
cfg.classif_adaptive_factor = False
cfg.classif_adaptive_min_factor = 0.
cfg.pod_deeplab_mask_factor = None
cfg.pod_interpolate_last = False
cfg.deeplab_mask_downscale = False
cfg.spp_scales = [1, 2, 4]
cfg.pod_apply = "all" #["all", "backbone", "deeplab"]
cfg.no_pod_schedule = False
cfg.use_pod_schedule = cfg.no_pod_schedule
cfg.pseudo_soft = None #["soft_certain", "soft_uncertain"]
cfg.pseudo_soft_factor = 1.0
cfg.pseudo_ablation = None #["corrected_errors", "removed_errors"]
cfg.ce_on_new = False
cfg.ce_on_pseudo = False
cfg.nca = False
cfg.nca_margin = 0.
cfg.entropy_min = 0.
cfg.pod_prepro = "pow"
cfg.use_cosine = False
cfg.nb_background_modes = 1
cfg.multimodal_fusion = "sum"
cfg.align_weight_frequency = "never"  #choices = ["never", "epoch", "task"]"]
cfg.align_weight = None #["old", "background", "all"]
cfg.base_weights = False
cfg.pseudo_nb_bins = None
cfg.step_threshold = None
cfg.sample_weights_new = None
cfg.pod_deeplab_mask = False
cfg.init_multimodal = None

cfg.freeze = False
cfg.unknown = False
cfg.pseudo_ssul = False
cfg.w_transfer = False
cfg.pseudo_thresh = 0.7


cfg.memory = True
cfg.shuffle = False
###############################
# TRAIN
cfg.TRAIN = edict()

cfg.TRAIN.FREEZE_AUDIO_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_VGGISH_MODEL_PATH = "./torchvggish/vggish-10086976.pth"
cfg.TRAIN.PREPROCESS_AUDIO_TO_LOG_MEL = True #! notice
cfg.TRAIN.POSTPROCESS_LOG_MEL_WITH_PCA = False
cfg.TRAIN.PRETRAINED_PCA_PARAMS_PATH = "./torchvggish/vggish_pca_params-970ea276.pth"
cfg.TRAIN.FREEZE_VISUAL_EXTRACTOR = True
cfg.TRAIN.PRETRAINED_RESNET50_PATH = "../../pretrained_backbones/resnet50-19c8e357.pth"
cfg.TRAIN.PRETRAINED_PVTV2_PATH = "../../pretrained_backbones/pvt_v2_b5.pth"

cfg.TRAIN.FINE_TUNE_SSSS = False
cfg.TRAIN.PRETRAINED_S4_AVS_WO_TPAVI_PATH = "../single_source_scripts/logs/ssss_20220118-111301/checkpoints/checkpoint_29.pth.tar"
cfg.TRAIN.PRETRAINED_S4_AVS_WITH_TPAVI_PATH = "../single_source_scripts/logs/ssss_20220118-112809/checkpoints/checkpoint_68.pth.tar"

###############################
# DATA
cfg.DATA = edict()
cfg.DATA.CROP_IMG_AND_MASK = True
cfg.DATA.CROP_SIZE = 224 # short edge

cfg.DATA.LABEL_IDX_PATH = "../../avsbench_data/AVSBench_semantic/label2idx.json" #! notice: you need to change the path
cfg.DATA.META_CSV_PATH = "../../avsbench_data/AVSBench_semantic/metadata.csv" #! notice: you need to change the path to metadata,metav2s, metav2m
cfg.DATA.DIR_BASE = "../../avsbench_data/AVSBench_semantic" #! notice: you need to change the path
cfg.DATA.MEMORY_CSV_PATH = "../../avsbench_data/AVSBench_semantic/expanded_memory_60_10.csv" #! notice: you need to change the path
# cfg.DATA.DIR_MASK = "../../avsbench_data/AVSBench_semantic/v2_data/gt_masks" #! notice: you need to change the path
# cfg.DATA.DIR_COLOR_MASK = "../../avsbench_data/avsv2_data/gt_color_masks_rgb" #! notice: you need to change the path
cfg.DATA.IMG_SIZE = (224, 224)
###############################
cfg.DATA.RESIZE_PRED_MASK = True
cfg.DATA.SAVE_PRED_MASK_IMG_SIZE = (360, 240) # (width, height)


# def _edict2dict(dest_dict, src_edict):
#     if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
#         for k, v in src_edict.items():
#             if not isinstance(v, edict):
#                 dest_dict[k] = v
#             else:
#                 dest_dict[k] = {}
#                 _edict2dict(dest_dict[k], v)
#     else:
#         return


# def gen_config(config_file):
#     cfg_dict = {}
#     _edict2dict(cfg_dict, cfg)
#     with open(config_file, 'w') as f:
#         yaml.dump(cfg_dict, f, default_flow_style=False)


# def _update_config(base_cfg, exp_cfg):
#     if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
#         for k, v in exp_cfg.items():
#             if k in base_cfg:
#                 if not isinstance(v, dict):
#                     base_cfg[k] = v
#                 else:
#                     _update_config(base_cfg[k], v)
#             else:
#                 raise ValueError("{} not exist in config.py".format(k))
#     else:
#         return


# def update_config_from_file(filename):
#     exp_config = None
#     with open(filename) as f:
#         exp_config = edict(yaml.safe_load(f))
#         _update_config(cfg, exp_config)

if __name__ == "__main__":
    print(cfg)