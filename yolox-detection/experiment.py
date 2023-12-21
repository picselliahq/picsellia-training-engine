from YOLOX.tools.train import main
from YOLOX.tools.train import make_parser
from YOLOX.yolox.core import launch
from YOLOX.yolox.exp import get_exp, check_exp_value
from YOLOX.yolox.utils import configure_module, get_num_devices

configure_module()
args = make_parser().parse_args()

args.name = "yolox-m"
args.batch_size = 4
args.epoch = 1
args.data_dir = (
    "/home/alexis/Downloads/yolox_dataset"  # Set the path to the training dataset
)

args.ckpt = (
    "/home/alexis/Downloads/yolox_m.pth"  # Set the path to the pre-trained weights file
)
args.resume = True

exp = get_exp(None, args.name)
exp.merge(args.opts)
check_exp_value(exp)

if not args.experiment_name:
    args.experiment_name = exp.exp_name

num_gpu = get_num_devices() if args.devices is None else args.devices
assert num_gpu <= get_num_devices()

if args.cache is not None:
    exp.dataset = exp.get_dataset(cache=True, cache_type=args.cache)

dist_url = "auto" if args.dist_url is None else args.dist_url
launch(
    main,
    num_gpu,
    args.num_machines,
    args.machine_rank,
    backend=args.dist_backend,
    dist_url=dist_url,
    args=(exp, args),
)
