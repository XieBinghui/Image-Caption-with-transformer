__author__ = 'tylin'
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from spice.spice import Spice

def evaluate(gts=None, res=None):
    # imgIds = self.coco.getImgIds()
    # =================================================
    # Set up scorers
    # =================================================
    print 'setting up scorers...'
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    # =================================================
    # Compute scores
    # =================================================
    res_scores = []
    for scorer, method in scorers:
        print 'computing %s score...'%(scorer.method())
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                res_scores.append((m, sc))
        else:
            res_scores.append((method, score))
    return res_scores
            
class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self, gts=None, res=None):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        if gts is None and res is None:
            gts = {}
            res = {}
            for imgId in imgIds:
                gts[imgId] = self.coco.imgToAnns[imgId]
                res[imgId] = self.cocoRes.imgToAnns[imgId]
            
            print("before")
            print(gts[184321])
            print(res[184321])

            # =================================================
            # Set up scorers
            # =================================================
            print 'tokenization...'
            tokenizer = PTBTokenizer()
            gts  = tokenizer.tokenize(gts)
            res = tokenizer.tokenize(res)
        
        print("after")
        return gts, res
        print(gts[184321])
        print(res[184321])

        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print "%s: %0.3f"%(m, sc)
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print "%s: %0.3f"%(method, score)
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
            
        for imgId, score in zip(imgIds, scores):
            if method == "CIDEr" and score==1:
                print imgId
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
