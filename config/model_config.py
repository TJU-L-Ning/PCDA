import os 
import argparse

def build_args():
    
    parser = argparse.ArgumentParser("This script is set for the PCDA under SF-UniDA in Remote Sensing Scene Classification")
    
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--backbone_arch", type=str, default="resnet50")
    parser.add_argument("--embed_feat_dim", type=int, default=256)
    parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--t_idx", type=int, default=0)
    parser.add_argument("--idex", default=None, type=int)
    parser.add_argument("--checkpoint", default="None", type=str)
    parser.add_argument("--src_flg", default=False, type=bool)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gpu", default='1', type=str)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=...)
    parser.add_argument("--weight_decay", type=float, default=...)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", default=..., type=int)
    parser.add_argument("--lamda", default=..., type=float)
    parser.add_argument("--cou", default=100, type=int)
    parser.add_argument("--rho", default=0.7, type=float)
    parser.add_argument("--source_train_type", default="smooth", type=str, help="vanilla, smooth")
    parser.add_argument("--target_label_type", default="OPDA", type=str)
    parser.add_argument("--target_private_class_num", default=None, type=int)
    parser.add_argument("--note", default="PCDA")
    parser.add_argument("--option", default="target")    
    args = parser.parse_args()
    if args.dataset == "R2U":
        args.source_data_dir = "/root/code/list/list-R2U"
        args.target_data_dir = "/root/code/list/list-R2U"
        args.source_domain_list = ["R7"]
        args.target_domain_list = ["UCM"]
        args.target_domain_dir_list = [args.target_data_dir]
        args.train_source_list = "R7_train.txt"
        args.train_target_list = "UCM.txt"
        args.save_checkpoints = "checkpoints_R2U_2-5-16"
        args.test_source_list = "R7_test.txt"
        args.test_target_list = "UCM.txt"
        args.shared_class_num = 5
        args.source_private_class_num = 2
        args.target_private_class_num = 16
 
    elif args.dataset == "A2N":
        args.source_data_dir = "/home/ubuntu/HDDs/HDD2/PCDA-main/list/list-A2N/"
        args.target_data_dir = "/home/ubuntu/HDDs/HDD2/PCDA-main/list/list-A2N/"
        args.source_domain_list = ["AID"]
        args.target_domain_list = ["NWPU"]
        args.target_domain_dir_list = [args.target_data_dir]
        args.train_source_list = "AID_train.txt"
        args.train_target_list = "NWPU_train.txt"
        args.save_checkpoints = "checkpoints_A2N_10-20-25"
        args.test_source_list = "AID_test.txt"
        args.test_target_list = "NWPU_test.txt"
        args.shared_class_num = 20
        args.source_private_class_num = 10
        args.target_private_class_num = 25  
    elif args.dataset == "R2A":
        args.source_data_dir = "/home/ubuntu/HDDs/HDD2/PCDA-main/list/list-R2A/"
        args.target_data_dir = "/home/ubuntu/HDDs/HDD2/PCDA-main/list/list-R2A/"
        args.source_domain_list = ["R7"]
        args.target_domain_list = ["AID"]
        args.target_domain_dir_list = [args.target_data_dir]
        args.train_source_list = "R7_train.txt"
        args.train_target_list = "AID.txt"
        args.save_checkpoints = "checkpoints_R2A_1-6-24"
        args.test_source_list = "R7_test.txt"
        args.test_target_list = "AID.txt"
        args.shared_class_num = 6
        args.source_private_class_num = 1
        args.target_private_class_num = 24     
    elif args.dataset == "R2N":
        args.source_data_dir = "/home/ubuntu/HDDs/HDD2/PCDA-main/list/list-R2N/"
        args.target_data_dir = "/home/ubuntu/HDDs/HDD2/PCDA-main/list/list-R2N/"
        args.source_domain_list = ["R7"]
        args.target_domain_list = ["NWPU"]
        args.target_domain_dir_list = [args.target_data_dir]
        args.train_source_list = "R7_train.txt"
        args.train_target_list = "NWPU_train.txt"
        args.save_checkpoints = "checkpoints_R2N_1-6-39"
        args.test_source_list = "R7_test.txt"
        args.test_target_list = "NWPU_test.txt"
        args.shared_class_num = 6
        args.source_private_class_num = 1
        args.target_private_class_num = 39   
        
    else:
            raise NotImplementedError("Unknown target label type specified", args.target_label_type)
    args.source_class_num = args.shared_class_num + args.source_private_class_num 
    args.target_class_num = args.shared_class_num + args.target_private_class_num 
    args.class_num = args.source_class_num
    args.source_class_list = [i for i in range(args.source_class_num)]
    args.target_class_list = [i for i in range(args.shared_class_num)] 
    if args.target_private_class_num > 0: 
        args.target_class_list.append(args.source_class_num)
    print(args)

    return args

