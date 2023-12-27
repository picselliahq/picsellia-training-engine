from YOLOX.tools.train import main
from YOLOX.tools.train import make_parser
from YOLOX.yolox.core import launch
from YOLOX.yolox.exp import get_exp_by_name, check_exp_value
from YOLOX.yolox.utils import configure_module, get_num_devices

# 1 - Args
configure_module()
args = make_parser().parse_args()

args.name = "yolox-s"
args.batch_size = 4
args.epochs = 1
args.data_dir = "/home/alexis/Downloads/yolox_dataset"
args.train_ann = f"{args.data_dir}/instances_train2017.json"
args.test_ann = f"{args.data_dir}/instances_test2017.json"
args.val_ann = f"{args.data_dir}/instances_val2017.json"
args.num_classes = 3
args.learning_rate = 1e-4

args.ckpt = "/home/alexis/Downloads/yolox_s.pth"

# 2 - Get model architecture
exp = get_exp_by_name(args)
exp.merge(args.opts)
check_exp_value(exp)

if not args.experiment_name:
    args.experiment_name = exp.exp_name

num_gpu = get_num_devices() if args.devices is None else args.devices
assert num_gpu <= get_num_devices()

# 3 - Launch training
launch(
    main_func=main,
    num_gpus_per_machine=num_gpu,
    args=(exp, args),
)
