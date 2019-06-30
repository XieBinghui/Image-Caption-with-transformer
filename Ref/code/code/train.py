import torch

from transformer.flow import make_model
from transformer.label_smoothing import LabelSmoothing, LossCompute
from transformer.noam_opt import NoamOpt

from torch.utils.data import DataLoader
from dataset import FlickrDataset, EOS, TRAIN, VALID, TEST

from opts import parse_opt
from inference import greedy_decode
from evaluate import evaluate_output

import time, os

args = parse_opt()

def run_epoch(data_iter, valid_data_iter, model, loss_compute, data):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0

    print(args)
    print(evaluate_output(valid_data_iter, args, model, loss_compute, data, VALID))
    CIDEr = 0

    for i, batch in enumerate(data_iter):
        model.train()
        src, trg = batch
        trg, trg_mask, trg_y, ntokens = data.make_target_mask_ntoken(trg)
        if args.is_cuda:
            src, trg, trg_mask, trg_y = src.cuda(), trg.cuda(), trg_mask.cuda(), trg_y.cuda()
        out = model.forward(src, trg, trg_mask)
        loss = loss_compute(out, trg_y, ntokens)
        total_loss += loss
        total_tokens += ntokens
        tokens += ntokens
        if i % 5 == 0:
            elapsed = time.time() - start
            print("[%.4f] Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (elapsed, i, loss / ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

            if i % 50 == 0:
                out = greedy_decode(model, src[:1, ...], 100, data).cpu()
                sentence = []
                for j in range(1, out.size(1)):
                    sym = data.idx2word[out[0, j].item()]
                    if sym == EOS: break
                    sentence.append(sym)

                model.eval()

                results = evaluate_output(valid_data_iter, args, model, loss_compute, data, VALID)
                for res in results:
                    print(res[0], res[1])
                if not os.path.exists(args.task):
                    os.makedirs(args.task)
                with open("%s/log.txt" % args.task, "a") as f:
                    f.write(str(results) + "\n")
                    f.write("loss: %.4f" % (loss / ntokens))
                print(" ".join(sentence))

                if results[-1][-1] > CIDEr:
                    CIDEr = results[-1][-1]

                    torch.save(model.state_dict(),
                           "%s/lr_%e_bs_%d.pkl" % (args.task, args.lr, args.batch_size))

    return total_loss / total_tokens

if __name__ == "__main__":

    data = FlickrDataset(args, data_type=TRAIN)
    train_loader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    data = FlickrDataset(args, data_type=VALID)
    valid_loader = DataLoader(dataset=data, batch_size=32, shuffle=False, drop_last=True)

    loss_compute = LabelSmoothing(data.vocab_size, data.padding, smoothing=0.0)
    model = make_model(data.vocab_size, n=3, d_model=args.d_model, d_ff=4*args.d_model, h=8, dropout=0.1, att=args.att, src_position=args.src_posi)
    print(model)
    if args.is_cuda:
        model.cuda()
        loss_compute = loss_compute.cuda()
    # model.load_state_dict(torch.load(args.restore))
    # print(" ==> restore from %s " % args.restore)
    # model_opt = NoamOpt(args.d_model, 1, 400, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    
    model_opt = NoamOpt(2048, 1, 500,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(10):
        model.train()
        run_epoch(train_loader, valid_loader, model,
                LossCompute(model.generator, loss_compute, model_opt), data)