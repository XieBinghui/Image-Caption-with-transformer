import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Data input settings

    parser.add_argument('--dataset', type=str, default="../DATASET_Flickr30k")

    parser.add_argument('--feature', type=str, default="resnet101_fea/fea_att"
    parser.add_argument('--cap_json', type=str, default="cap_flickr30k.json")

    parser.add_argument('--dic_json', type=str, default="dic_flickr30k.json")

    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')

    parser.add_argument('--d_model', type=int, default=2048, help='dimension of model')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

    parser.add_argument('--max_len', type=int, default=90, help='max length')

    parser.add_argument('--is_cuda', type=int, default=1, help='if use cuda')

    parser.add_argument('--task', type=str, default="task", help='task name for ckpt')

    parser.add_argument('--att', type=int, default=1, help='self-att for encoder')

    parser.add_argument('--src_posi', type=int, default=0, help='add postional encoding on src matrix')

    parser.add_argument('--restore', type=str, default="../att_1/lr_5.000000e-03_bs_32.pkl")

    return parser.parse_args()