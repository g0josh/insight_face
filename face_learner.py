
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
import torch
from torchvision import transforms as trans


class face_learner(object):
    def __init__(self, conf, inference=False):
        self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
        print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        self.threshold = conf.threshold

    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path
        print ("save path = {}".format(save_path))
        self.model.load_state_dict(torch.load('/home/cbarobotics/dev/InsightFace_Pytorch/work_space/save/model_{}'.format(fixed_str)))

    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)

        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum