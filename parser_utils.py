

def parser_add_model_argument(parser):
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)

    # partition contrastive learning
    parser.add_argument('--n_trans', type=int, default=11)
    parser.add_argument('--partition_num', type=int, default=20)
    parser.add_argument('--anchor_partition', type=list, default=None)
    parser.add_argument('--center_type', type=str, default='Mean')
    parser.add_argument('--act', type=str, default='LeakyReLU')
    parser.add_argument('--rep_dim', type=int, default=64)
    parser.add_argument('--lamda', type=float, default=1)

    return parser


def update_model_configs(args, model_configs):

    if args.model.startswith('pclad'):
        model_configs['epochs'] = args.epochs
        model_configs['lr'] = args.lr
        model_configs['anchor_partition'] = args.anchor_partition
        model_configs['center_type'] = args.center_type
        model_configs['partition_num'] = args.partition_num
        model_configs['n_trans'] = args.n_trans
        model_configs['rep_dim'] = args.rep_dim
        model_configs['act'] = args.act
        model_configs['lamda'] = args.lamda

    return model_configs