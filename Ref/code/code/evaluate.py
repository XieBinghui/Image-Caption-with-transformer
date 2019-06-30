import time, os, json
from dataset import EOS, IMG_INFO, TRAIN, VALID, TEST, CAPTION, IMG_INFO_SPLIT, IMG_INFO_IDX

from inference import greedy_decode
from caption.pycocoevalcap.eval import evaluate
from tqdm import tqdm
import collections

def evaluate_output(data_iter, args, model, loss_compute, data, data_type):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    print("Evaluation...")

    cap_json = json.load(open(os.path.join(args.dataset, args.cap_json)))
    dic_json = json.load(open(os.path.join(args.dataset, args.dic_json)))[IMG_INFO]

    gts_backup, res = {}, {}

    for cap, dic in zip(cap_json, dic_json):
        if dic[IMG_INFO_SPLIT] == data_type:
            str_idx = dic[IMG_INFO_IDX]
            gts_backup[int(str_idx)] = [" ".join(c[CAPTION]) for c in cap]

    for i, batch in tqdm(enumerate(data_iter)):
        str_idx, src, trg = batch
        str_idx = [int(idx) for idx in str_idx]

        trg, trg_mask, trg_y, ntokens = data.make_target_mask_ntoken(trg)

        if args.is_cuda:
            src, trg, trg_mask, trg_y = src.cuda(), trg.cuda(), trg_mask.cuda(), trg_y.cuda()

        out = greedy_decode(model, src, 90, data).cpu()
        for idx, o in enumerate(out):
            sym = []
            for j in range(1, o.size(0)):
                sym_ = data.idx2word[o[j].item()]
                sym.append(sym_)
                if sym_ == EOS: break

            res[str_idx[idx]] = [" ".join(sym)]

    gts = collections.OrderedDict()

    for k in res:
        gts[k] = gts_backup[k]

    print(len(gts.keys()))
    print(len(res.keys()))

    return evaluate(gts, res)
