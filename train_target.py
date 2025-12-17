import os
import faiss
import torch 
import shutil 
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 
from model.PCDA import SFUniDA
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader

from config.model_config import build_args
from utils.utils import set_logger, set_random_seed
from utils.utils import get_acc, CrossEntropyLabelSmooth

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE


def temperature_scaling_softmax(logits, temperature=1.0):
    scaled_logits = logits / temperature
    probabilities = F.softmax(scaled_logits, dim=-1)
    return probabilities

def kl_divergence_loss(predicted_probs, true_labels):
    kl_loss = F.kl_div(torch.log(predicted_probs + 1e-10), true_labels, reduction='sum')
    return kl_loss


from Loss.loss import ProtoLoss





def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

best_score = 0.0
best_coeff = 1.0


@torch.no_grad()
def obtain_global_pseudo_labels(args, model, dataloader, epoch_idx=0.0):
    model.eval()

    pred_cls_bank = []
    gt_label_bank = []
    embed_feat_bank = [] 
    pos_topk_num_bank=[] 
    class_list = args.target_class_list 
    
   
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        imgs_test = imgs_test.cuda()
        embed_feat, pred_cls = model(imgs_test, apply_softmax=True)
        pred_cls_bank.append(pred_cls)
        embed_feat_bank.append(embed_feat)
        gt_label_bank.append(imgs_label.cuda())
    
    pred_cls_bank = torch.cat(pred_cls_bank, dim=0)
    gt_label_bank = torch.cat(gt_label_bank, dim=0) 
    embed_feat_bank = torch.cat(embed_feat_bank, dim=0) 
    embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)
    
    global best_score, best_coeff

    if epoch_idx == 0.0:
        embed_feat_bank_cpu = embed_feat_bank.cpu().numpy()
        embed_feat_bank_cpu = TSNE(n_components=2, init="pca", random_state=0).fit_transform(embed_feat_bank_cpu)

        coeff_list = [0.25, 0.50, 1, 1.5, 2, 2.5, 3]
        for coeff in coeff_list:
            KK = max(int(args.class_num * coeff), 2)
            kmeans = KMeans(n_clusters=KK, random_state=0).fit(embed_feat_bank_cpu)
            cluster_labels = kmeans.labels_
            sil_score = silhouette_score(embed_feat_bank_cpu, cluster_labels)

            if sil_score > best_score:
                best_score = sil_score
                best_coeff = coeff
    
    KK = int(args.class_num * best_coeff)
    data_num = pred_cls_bank.shape[0]

  
    top2_vals, top2_idxs = torch.topk(pred_cls_bank, k=2, dim=1, largest=True)
    max_vals = top2_vals[:, 0]
    second_max_vals = top2_vals[:, 1]
    pred_classes = top2_idxs[:, 0]
    sample_diffs = max_vals - second_max_vals
    avg_diff_list = [0.0 for _ in range(args.class_num)]
    class_sample_count = torch.zeros(args.class_num, device=pred_cls_bank.device)
    
    for cls_idx in range(args.class_num):
        cls_mask = (pred_classes == cls_idx)
        cls_diffs = sample_diffs[cls_mask]
        if len(cls_diffs) > 0:  
            avg_diff_list[cls_idx] = cls_diffs.mean().item()  
            class_sample_count[cls_idx] = len(cls_diffs)

    sorted_pred_cls, sorted_pred_cls_idxs = torch.sort(pred_cls_bank, dim=0, descending=True) 

    pos_topk_idxs_dict = []
    neg_topk_idxs_dict = []

    
    for i in range(args.class_num):
        base = data_num / KK
        dynamic_coeff = 0.1 + 0.9 * (avg_diff_list[i] / 2)**0.5 if avg_diff_list[i] >=0 else 0.1
        pos_topk_num = int(base * dynamic_coeff)
        pos_topk_num = max(pos_topk_num, 1)  
        pos_topk_num_bank.append(pos_topk_num)
        
        pos_idxs = sorted_pred_cls_idxs[:pos_topk_num, i] 
        neg_idxs = sorted_pred_cls_idxs[pos_topk_num:, i]  
        
        
        pos_idxs = pos_idxs.unsqueeze(1).expand(-1, args.embed_feat_dim) 
        neg_idxs = neg_idxs.unsqueeze(1).expand(-1, args.embed_feat_dim) 
        
        pos_topk_idxs_dict.append(pos_idxs)
        neg_topk_idxs_dict.append(neg_idxs)


    max_topk = max(pos_topk_num_bank)
    pos_topk_idxs = torch.zeros((args.class_num, max_topk, args.embed_feat_dim), 
                                device=pred_cls_bank.device, dtype=torch.long)
    
    for cls_idx in range(args.class_num):
        pos_topk_num = pos_topk_num_bank[cls_idx]
        pos_topk_idxs[cls_idx, :pos_topk_num, :] = pos_topk_idxs_dict[cls_idx]

    embed_feat_bank_expand = embed_feat_bank.unsqueeze(0).expand([args.class_num, -1, -1]) 
    pos_feat_sample = []
    
    for cls_idx in range(args.class_num):
        topk_num = pos_topk_num_bank[cls_idx]
        cls_pos_idxs = pos_topk_idxs[cls_idx, :topk_num, :].unsqueeze(0)
        cls_pos_feat = torch.gather(embed_feat_bank_expand[cls_idx:cls_idx+1], 1, cls_pos_idxs) 
        pad_size = max_topk - topk_num
        if pad_size > 0:
            cls_pos_feat = torch.cat([
                cls_pos_feat,
                torch.zeros((1, pad_size, args.embed_feat_dim), device=cls_pos_feat.device)
            ], dim=1)
        
        pos_feat_sample.append(cls_pos_feat)

    pos_feat_sample = torch.cat(pos_feat_sample, dim=0)

    
    pos_cls_prior = []
    for cls_idx in range(args.class_num):
        topk_num = pos_topk_num_bank[cls_idx]
        cls_sorted_pred = sorted_pred_cls[:topk_num, cls_idx].mean()
        cls_prior = cls_sorted_pred * (1.0 - args.rho) + args.rho
        pos_cls_prior.append(cls_prior)
    pos_cls_prior = torch.tensor(pos_cls_prior, device=pred_cls_bank.device).unsqueeze(1)  
    
    pos_feat_proto = []
    for cls_idx in range(args.class_num):
        topk_num = pos_topk_num_bank[cls_idx]
        cls_pos_feat = pos_feat_sample[cls_idx:cls_idx+1, :topk_num, :] 
        cls_pos_proto = 0.7 * torch.mean(cls_pos_feat, dim=1, keepdim=True)
        pos_feat_proto.append(cls_pos_proto)
    pos_feat_proto = torch.cat(pos_feat_proto, dim=0)  #[C, 1, D]
    pos_feat_proto = pos_feat_proto / torch.norm(pos_feat_proto, p=2, dim=-1, keepdim=True)
  
    feat_proto_pos_simi = torch.zeros((data_num, args.class_num), device=pred_cls_bank.device)
    feat_proto_max_simi = torch.zeros((data_num, args.class_num), device=pred_cls_bank.device)
    feat_proto_max_idxs = torch.zeros((data_num, args.class_num), device=pred_cls_bank.device, dtype=torch.long)
    
    all_neg_protos = [] 
    for cls_idx in range(args.class_num):
    
        neg_feat_cls_sample = torch.gather(embed_feat_bank, 0, neg_topk_idxs_dict[cls_idx]) 
        neg_feat_cls_sample_np = neg_feat_cls_sample.cpu().numpy()
        
     
        faiss_kmeans = faiss.Kmeans(args.embed_feat_dim, KK, niter=100, verbose=False, min_points_per_centroid=1, gpu=False)
        faiss_kmeans.train(neg_feat_cls_sample_np)
        
        cls_neg_feat_proto = torch.from_numpy(faiss_kmeans.centroids).cuda()  
        cls_neg_feat_proto = cls_neg_feat_proto / torch.norm(cls_neg_feat_proto, p=2, dim=-1, keepdim=True)
        all_neg_protos.append(cls_neg_feat_proto)
        
      
        cls_pos_feat_proto = pos_feat_proto[cls_idx, :] 
        cls_pos_simi = torch.einsum("nd, kd -> nk", embed_feat_bank, cls_pos_feat_proto) 
        cls_neg_simi = torch.einsum("nd, kd -> nk", embed_feat_bank, cls_neg_feat_proto) 
        
        cls_pos_simi = cls_pos_simi * pos_cls_prior[cls_idx]
        cls_simi = torch.cat([cls_pos_simi, cls_neg_simi], dim=1)  
        
        feat_proto_pos_simi[:, cls_idx] = cls_simi[:, 0]
        maxsimi, maxidxs = torch.max(cls_simi, dim=-1)
        feat_proto_max_simi[:, cls_idx] = maxsimi
        feat_proto_max_idxs[:, cls_idx] = maxidxs

   
    pos_prototypes = pos_feat_proto.view(args.class_num, args.embed_feat_dim)
    
    all_feat_protos = torch.cat((all_neg_protos[1], pos_prototypes), dim=0)

    


    psd_label_prior_simi = torch.einsum("nd, cd -> nc", embed_feat_bank, pos_feat_proto.squeeze(1))
    psd_label_prior_idxs = torch.max(psd_label_prior_simi, dim=-1, keepdim=True)[1]
    psd_label_prior = torch.zeros_like(psd_label_prior_simi).scatter(1, psd_label_prior_idxs, 1.0)
    
    hard_psd_label_bank = (feat_proto_max_idxs == 0).float()
    hard_psd_label_bank = hard_psd_label_bank * psd_label_prior
    
    hard_label = torch.argmax(hard_psd_label_bank, dim=-1)
    hard_label_unk = (torch.sum(hard_psd_label_bank, dim=-1) == 0)  

    hard_label[hard_label_unk] = args.class_num
    
    hard_psd_label_bank[hard_label_unk, :] += 1.0
    hard_psd_label_bank = hard_psd_label_bank / (torch.sum(hard_psd_label_bank, dim=-1, keepdim=True) + 1e-4)
    hard_psd_label_bank = hard_psd_label_bank.cuda()

    known_indices = torch.nonzero(~hard_label_unk).squeeze(1)
    known_samples_features = embed_feat_bank[known_indices]

    per_class_num = np.zeros((len(class_list)))
    pre_class_num = np.zeros_like(per_class_num)
    per_class_correct = np.zeros_like(per_class_num)
    
    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_label_bank == label)[0]
        correct_idx = torch.where(hard_label[label_idx] == label)[0]
        pre_class_num[i] = float(len(torch.where(hard_label == label)[0]))
        per_class_num[i] = float(len(label_idx))
        per_class_correct[i] = float(len(correct_idx))
    
    per_class_acc = per_class_correct / (per_class_num + 1e-5)
    
    args.logger.info("PSD AVG ACC:\t" + "{:.3f}".format(np.mean(per_class_acc)))
    args.logger.info("PSD PER ACC:\t" + "\t".join(["{:.3f}".format(item) for item in per_class_acc]))
    args.logger.info("PER CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_num]))
    args.logger.info("PRE CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in pre_class_num]))
    args.logger.info("PRE ACC NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_correct]))
    return hard_psd_label_bank, pred_cls_bank, embed_feat_bank, pos_feat_proto, all_feat_protos , known_samples_features

@torch.no_grad()
def test(args, model, dataloader, src_flg=False):
    
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []
    
    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0
    
    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda() 
        _, pred_cls = model(imgs_test, apply_softmax=True)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())
    
    gt_label_all = torch.cat(gt_label_stack, dim=0) 
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) 

    h_score, known_acc, unknown_acc, _ = get_acc(args, class_list, gt_label_all, pred_cls_all, open_flg)
    return h_score, known_acc, unknown_acc



def train(args, model, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0):
       
    model.eval()
    hard_psd_label_bank, pred_cls_bank, embed_feat_bank, prototypes, _ , known_samples_features  = obtain_global_pseudo_labels(args, model, test_dataloader, epoch_idx) 

    model.train()

    prototypes.requires_grad_()


    proto_criterion = ProtoLoss(nav_t=1.0).cuda()

    

    all_pred_loss_stack = []
    psd_pred_loss_stack = []
    soft_label_loss_stack = []
    proto_loss_stack =[]
    
    iter_idx = epoch_idx * len(train_dataloader)  
    iter_max = args.epochs * len(train_dataloader)
    



    for imgs_train, _, _, imgs_idx in tqdm(train_dataloader, ncols=60):
        
        iter_idx += 1
        imgs_idx = imgs_idx.cuda()
        imgs_train = imgs_train.cuda()
        
        hard_psd_label = hard_psd_label_bank[imgs_idx] #[B, C]
        
        
        embed_feat, pred_cls = model(imgs_train, apply_softmax=True)

        soft_labels = temperature_scaling_softmax(pred_cls, temperature=2.0)
 
        hard_psd_label = hard_psd_label.float()


        psd_pred_loss = torch.sum(-hard_psd_label * torch.log(pred_cls + 1e-5), dim=-1).mean()

        soft_label_loss = kl_divergence_loss(soft_labels,hard_psd_label)

        proto_loss = proto_criterion(prototypes, embed_feat)
        loss = args.lamda* psd_pred_loss + proto_loss + soft_label_loss



        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_pred_loss_stack.append(loss.cpu().item())
        psd_pred_loss_stack.append(psd_pred_loss.cpu().item())
        soft_label_loss_stack.append(soft_label_loss.cpu().item())
        proto_loss_stack.append(proto_loss.cpu().item())
        
    train_loss_dict = {}
    train_loss_dict["all_pred_loss"] = np.mean(all_pred_loss_stack)
    train_loss_dict["psd_pred_loss"] = np.mean(psd_pred_loss_stack)
    train_loss_dict["soft_label_loss"] = np.mean(soft_label_loss_stack)
    train_loss_dict["proto_loss"] = np.mean(proto_loss_stack)
            
    return train_loss_dict
    

def main(args):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    
    model = SFUniDA(args)

    
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("checkpoint 路径有问题 检查checkpoint!!!")



    
    model = model.cuda()
    save_dir = os.path.join(this_dir, args.save_checkpoints, args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, args.note)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_target_training.txt") 
    
    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}] 
    
    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    for k, v in model.class_layer.named_parameters():  
        v.requires_grad = False  
        
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)



    target_data_list = open(os.path.join(args.target_data_dir, args.train_target_list), "r").readlines()
    target_dataset = SFUniDADataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)



    
    target_train_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False) ##lyt----
    
    notation_str =  "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="
    
    args.logger.info(notation_str)
    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0


    loss_stack = []
    
    for epoch_idx in tqdm(range(args.epochs), ncols=60):

        loss_dict =train(args, model, target_train_dataloader, target_test_dataloader, optimizer, epoch_idx)
        args.logger.info("Epoch: {}/{},          train_all_loss:{:.3f},\n\
                          train_psd_loss:{:.3f}, train_soft_label_loss:{:.3f}, train_proto_loss:{:.3f}, ".format(epoch_idx+1, args.epochs,
                                        loss_dict["all_pred_loss"], loss_dict["psd_pred_loss"], loss_dict["soft_label_loss"], loss_dict["proto_loss"] ))
        

        loss_stack.append(loss_dict["all_pred_loss"])


        hscore, knownacc, unknownacc = test(args, model, target_test_dataloader, src_flg=False)

        args.logger.info("Current: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(hscore, knownacc, unknownacc))
        percentage.append(knownacc)
        if (epoch_idx + 1) % 20 == 0:
            checkpoint_file_one = "epoch_{}_model.pth".format(epoch_idx + 1)
            torch.save({
                "epoch": epoch_idx,
                "model_state_dict": model.state_dict()}, os.path.join(save_dir, checkpoint_file_one))


        if knownacc >= best_known_acc:
            if hscore >= best_h_score:
                best_h_score = hscore
                best_known_acc = knownacc
                best_unknown_acc = unknownacc
                best_epoch_idx = epoch_idx
                
                checkpoint_file = "{}_SFDA_best_target_checkpoint_{}.pth".format(args.dataset,args.idex)         
                torch.save({
                        "epoch":epoch_idx,
                        "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
        
        args.logger.info("Best   : H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(best_h_score, best_known_acc, best_unknown_acc))
            
if __name__ == "__main__":
    args = build_args()
    set_random_seed(args.seed)
    
    percentage = []
    mk_stack = []
    args.checkpoint = os.path.join(args.save_checkpoints, args.dataset, "source_{}".format(args.s_idx),"source_{}_{}".format(args.source_train_type, args.target_label_type),"latest_source_checkpoint.pth")
    
    main(args)
    


