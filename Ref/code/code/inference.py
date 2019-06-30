import time, os, json, collections
import torch

from torch.utils.data import DataLoader
from dataset import FlickrDataset, BOS, EOS, IMG_INFO, TRAIN, VALID, TEST, CAPTION, IMG_INFO_SPLIT, IMG_INFO_IDX
from opts import parse_opt

from transformer.functional import subsequent_mask
from transformer.flow import make_model
from transformer.label_smoothing import LabelSmoothing, LossCompute
from tqdm import tqdm

args = parse_opt()

from caption.pycocoevalcap.eval import evaluate

def run_epoch(data_iter, model, loss_compute, data, data_type):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    print(args)

    gts, gts_backup, res = {}, {}, {}

    model.eval()

    cap_json = json.load(open(os.path.join(args.dataset, args.cap_json)))
    dic_json = json.load(open(os.path.join(args.dataset, args.dic_json)))[IMG_INFO]


    for cap, dic in zip(cap_json, dic_json):
        if dic[IMG_INFO_SPLIT] == data_type:
            str_idx = dic[IMG_INFO_IDX]
            gts_backup[int(str_idx)] = [" ".join(c[CAPTION]) for c in cap]


    for i, batch in tqdm(enumerate(data_iter)):
        str_idx, src, trg = batch
        str_idx = int(str_idx[0])
        # print(str_idx)

        trg, trg_mask, trg_y, ntokens = data.make_target_mask_ntoken(trg)

        if args.is_cuda:
            src, trg, trg_mask, trg_y = src.cuda(), trg.cuda(), trg_mask.cuda(), trg_y.cuda()

        out = greedy_decode(model, src, 100, data).cpu()

        sym = []
        for j in range(1, out.size(1)):
            sym_ = data.idx2word[out[0, j].item()]
            sym.append(sym_)
            if sym_ == EOS: break
        # print("answer...")
        # print(" ".join(sym))

        res[str_idx] = [" ".join(sym)]

    gts = collections.OrderedDict()

    for k in res:
        gts[k] = gts_backup[k]

    print(len(gts.keys()))
    print(len(res.keys()))

    print(evaluate(gts, res))

def greedy_decode(model, src, max_len, data):
    memory = model.encode(src)
    ys = torch.ones(src.size(0), 1).fill_(data.word2idx[BOS]).long()
    for i in range(max_len-1):
        ys_mask_ = subsequent_mask(ys.size(1)).cpu().numpy().astype('uint8')[0]
        ys_mask = torch.Tensor([ys_mask_ for _ in range(src.size(0))])
        if args.is_cuda:
            ys_mask, ys = ys_mask.cuda(), ys.cuda()
        out = model.decode(memory,
                           ys,
                           ys_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        ys = torch.cat([ys, next_word.long().unsqueeze(-1)], dim=1)
    return ys


if __name__ == "__main__":
    data_type = VALID
    data = FlickrDataset(args, data_type=data_type)
    data_loader = DataLoader(dataset=data, batch_size=1, shuffle=True, num_workers=16, drop_last=True)

    loss_compute = LabelSmoothing(data.vocab_size, data.padding)
    model = make_model(data.vocab_size, n=3, d_model=args.d_model, d_ff=4*args.d_model, h=8, dropout=0.1, att=args.att)
    print(model)

    if args.is_cuda:
        model.cuda()
        loss_compute = loss_compute.cuda()

    model.load_state_dict(torch.load(args.restore))
    print(" ==> restore from %s " % args.restore)

    model.eval()
    run_epoch(data_loader, model,
              LossCompute(model.generator, loss_compute, opt=None), data, data_type)
