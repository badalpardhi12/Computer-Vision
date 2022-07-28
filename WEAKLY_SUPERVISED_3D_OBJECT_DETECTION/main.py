
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import  WSDNN_Resnet
from dataset import KITTICam
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.ops import nms
from post_processing import calculate_ap, mean_average_precision, tensor_to_PIL, plot_3d
import wandb
import math
import sklearn
import sklearn.metrics
from loss import FocalLoss
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt


def train(train_loader, 
          model, 
          loss_fn,
          optimizer, 
          test_loader,
          num_classes=2):
    loss_bce_total = 0.0
    loss_total = 0.0
    data_count = 0.0
    total_target = torch.zeros((0, num_classes)).cuda()
    total_preds = torch.zeros((0, num_classes)).cuda()
    for iter, data in tqdm(enumerate(train_loader),
                           total=len(train_loader),
                           leave=False):
        model = model.train()
        bev = data['image'].cuda()
        labels = data['labels'].float().cuda()
        #gt_boxes = data['gt_boxes'].cuda()
        proposals = data['proposals'].squeeze().float().cuda()
        proposals = torch.cuda.FloatTensor(proposals)
        #gt_class_list = data['gt_class_list'].cuda()
        #with torch.cuda.amp.autocast():
        preds = model(bev, proposals)
        preds_class = preds.sum(dim=0).reshape(1, -1)
        #print(preds_class, labels)
        # preds_class_sigmoid = torch.sigmoid(preds_class)
        # total_preds = torch.cat([total_preds, preds_class_sigmoid], dim=0)
        # total_target = torch.cat([total_target, labels], dim=0)
        preds_class = torch.clamp(preds_class, 0, 1)
        #print(labels.shape, preds_class.shape, type(labels), type(preds_class))
        loss = loss_fn(preds_class, labels)
        
        bce_funtion = torch.nn.BCELoss(reduction='sum')
        loss_bce_total += bce_funtion(preds_class, labels).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        #loss_bce_total += F.binary_cross_entropy_with_logits(preds_class, labels, reduction='sum').item()
        loss_total += loss.item() * bev.shape[0]
        data_count += bev.shape[0]
        if iter%500 == 0 and iter != 0:
            #map_class = map_classification(total_preds, total_target)
            wandb.log({"Loss":loss_total / data_count})
            print("Focal Loss: ", loss_total / data_count , " BCE loss: ", loss_bce_total / data_count) #,  " mAP: ", map_class)

    return loss_bce_total / data_count

def validate(test_loader, 
             model, 
             loss_fn, 
             score_threshold=0.4,
             nms_iou_threshold=0.1,
             iou_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
             inv_class=None,
             direct_class=None,
             dataset=None):
    np.random.seed(2)
    num_classes = 1
    loss_total = 0.0
    data_count = 0.0
    all_gt_boxes_list = []
    all_pred_boxes_list = []
    all_gt_boxes = torch.zeros((0, 6))
    all_pred_boxes = torch.zeros((0, 7))
    plotting_idxs = np.random.randint(0, 500, (50))
    plotting_idxs_3d = plotting_idxs[:50]
    with torch.no_grad():
        for iter, data in tqdm(enumerate(test_loader),
                            total=len(test_loader),
                            leave=False):
            plotting_proposals = torch.zeros((0, 5))
            plotting_gts = torch.zeros((0, 5))
            bev = data['image'].cuda()
            filename = data['filename'][0]
            labels = data['labels'].float().cuda()
            gt_boxes = data['gt_boxes'].reshape(-1, 4) #.cuda()
            proposals = data['proposals'].squeeze().float().cuda()
            gt_class_list = data['gt_class_list'].reshape(-1) #.cuda()
            gt_boxes_3d = data['gt_boxes_3d'].reshape(-1, 6)
            proposals_3d = data['proposals_3d'].squeeze().float().cuda()
            scan = data['scan'].reshape(-1, 4)
            cls_probs= model(bev, proposals)
            data_count += bev.shape[0]

            for i in range(gt_boxes.shape[0]):
                modified_gt_box_list = [iter, 
                                        gt_class_list[i].item(),
                                        1,
                                        *gt_boxes[i].tolist()]
                all_gt_boxes_list.append(modified_gt_box_list)
                modified_boxes = torch.cat([torch.tensor([iter, gt_class_list[i]]), gt_boxes[i]]).reshape(1, -1)
                all_gt_boxes = torch.cat([all_gt_boxes, modified_boxes], dim=0)
                plotting_gts = torch.cat([plotting_gts,
                                          modified_boxes[0, 1:].reshape(1, -1)], dim=0)

            for class_num in range(num_classes):
                #curr_class_scores = bbox_scores[:, class_num]
                #### to plot pred in 3d offline plot
                #valid_3d_idx = torch.arange(0, cls_probs.shape[0])
                ####

                curr_class_scores = cls_probs[:, class_num]
                valid_score_idx = torch.where(curr_class_scores >= score_threshold / proposals.shape[0])
                valid_scores = curr_class_scores[valid_score_idx]
                
                #valid_3d_idx = valid_3d_idx[valid_score_idx[0]]

                valid_proposals = proposals[valid_score_idx[0], :]
                valid_proposals_3d = proposals_3d[valid_score_idx[0], :]
                retained_idx = nms(valid_proposals, valid_scores, nms_iou_threshold)
                retained_scores = valid_scores[retained_idx]
                retained_proposals = valid_proposals[retained_idx]
                retained_proposals_3d = valid_proposals_3d[retained_idx]
                
                #valid_3d_idx = valid_3d_idx[retained_idx]
                if iter in plotting_idxs_3d:
                    plot_3d(gt_boxes_3d, retained_proposals_3d, scan)
                #write_prediction(valid_3d_idx, filename)
                # curr_class_scores = cls_probs[:, class_num].squeeze(-1)
                # retained_idx = nms(proposals, curr_class_scores, nms_iou_threshold)
                # retained_scores = curr_class_scores[retained_idx]
                # retained_proposals = proposals[retained_idx]

                class_num_for_plotting = torch.ones((retained_proposals.shape[0], 1)) * class_num
                plotting_proposals = torch.cat([plotting_proposals,
                                                torch.cat([retained_proposals.detach().cpu(), 
                                                           class_num_for_plotting], dim=1)], dim=0)

                for i in range(retained_proposals.shape[0]):
                    modified_proposal_list = [iter,
                                              class_num,
                                              retained_scores[i].item(),
                                              *retained_proposals[i].detach().cpu().tolist()]
                    all_pred_boxes_list.append(modified_proposal_list)
                    modified_pred_boxes = torch.cat([torch.tensor([iter, class_num, retained_scores[i]]), 
                                                                retained_proposals[i].detach().cpu()]).reshape(1, -1)
                    all_pred_boxes = torch.cat([all_pred_boxes, modified_pred_boxes], dim=0)

            if iter in plotting_idxs:
                all_boxes = []
                all_gt_plotting_boxes = []
                raw_image = tensor_to_PIL(bev[0].detach().cpu())

                for idx in range(plotting_proposals.shape[0]):
                    box_data = {"position": {
                        "minX": plotting_proposals[idx, 0].item() / dataset.req_img_size[0],
                        "minY": plotting_proposals[idx, 1].item() / dataset.req_img_size[1],
                        "maxX": plotting_proposals[idx, 2].item() / dataset.req_img_size[0],
                        "maxY": plotting_proposals[idx, 3].item() / dataset.req_img_size[1]},
                        #"class_id": int(plotting_proposals[idx, 4].item()),
                        #"box_caption": inv_class[int(plotting_proposals[idx][4])],
                        "class_id": 1,
                        "box_caption": "Prediction Cars boxes",
                        }
                    all_boxes.append(box_data)
                

                for idx in range(plotting_gts.shape[0]):
                    box_data_new = {"position": {
                        "minX": plotting_gts[idx, 1].item() / dataset.req_img_size[0],
                        "minY": plotting_gts[idx, 2].item() / dataset.req_img_size[1],
                        "maxX": plotting_gts[idx, 3].item() / dataset.req_img_size[0],
                        "maxY": plotting_gts[idx, 4].item() / dataset.req_img_size[1]},
                        "class_id": 0,
                        "box_caption": "GT Cars boxes",
                        # "class_id": int(plotting_gts[idx, 0].item()),
                        # "box_caption": inv_class[int(plotting_gts[idx][0])],
                        }
                    all_gt_plotting_boxes.append(box_data_new)
                    
                box_image = wandb.Image(raw_image, 
                                        boxes={"predictions":
                                        {"box_data": all_boxes,
                                        "class_labels": inv_class},
                                             "ground_truth":
                                        {"box_data": all_gt_plotting_boxes,
                                        "class_labels": inv_class}
                                        })
                wandb.log({"Image proposals " + data['filename'][0]: box_image})
                # box_image = wandb.Image(raw_image, 
                #                         boxes= {"predictions":
                #                         {"box_data": all_gt_plotting_boxes,
                #                         "class_labels": inv_class}
                #                         })
                # wandb.log({"Image gt " + str(iter): box_image})
    xss = []
    yss = []
    keys = []           
    for iou in iou_list:
        #print(all_gt_boxes.shape, all_gt_boxes.shape)
        ####
        his_ap = mean_average_precision(all_pred_boxes_list, 
                                        all_gt_boxes_list,
                                        iou)
        print("His ap: ", his_ap)
        ####

        AP, xs, ys = calculate_ap(all_pred_boxes, all_gt_boxes, iou, inv_class=inv_class, total_cls_num=num_classes)
        mAP = 0 if len(AP) == 0 else sum(AP) / len(AP)
        #return mAP.item(), AP
        wandb.log({"map@ " + str(iou): mAP})
        print("Iou ", iou, " mAP ", mAP)
        xss.append(xs)
        yss.append(ys)
        keys.append(str(iou))

    fig, ax = plt.subplots(1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    for no in range(len(xss)):
        ax.plot(xss[no], yss[no], label=keys[no])
    ax.legend()
    wandb.log({"PR_org": fig})
    image_pr = wandb.Image(fig)
    wandb.log({"PR_matplotlib": image_pr})
    wandb.log({"PR-Curves": wandb.plot.line_series(xss,
                                                   yss,
                                                   keys=keys)})

    return mAP


if __name__ == '__main__':
    #torch.random.seed(100)
    torch.manual_seed(100)
    valid_data_list_filename = "./valid_full_list.txt"
    lidar_folder_name = "./data/"
    dataset = KITTICam(valid_data_list_filename=valid_data_list_filename, 
                       lidar_folder_name=lidar_folder_name,
                       req_img_size=(1024, 512))
    wandb.init(project="WSDNN_Resnet")
    epochs = 50
    #model = WSDNN_Alexnet(roi_size=(12, 6))
    model = WSDNN_Resnet()
    
    train_dataset_length = int(0.80 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_dataset_length,
                                                        len(dataset) - train_dataset_length],
                                                        generator=torch.Generator().manual_seed(100))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(len(train_dataset), len(test_dataset))

    #loss_fn = nn.BCELoss(reduction='sum')
    loss_fn = FocalLoss(alpha=0.25, gamma=2)

    model = model.cuda()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer.load_state_dict(torch.load('opt_res_focal_loss_plt.pth'))
    model.load_state_dict(torch.load('model_res_focal_loss_plt.pth'))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    for i in range(epochs):
        if i%1 == 0:
            model = model.eval()
            mAP = validate(test_loader, 
                          model, 
                          loss_fn, 
                          inv_class=dataset.inv_class, 
                          direct_class=dataset.class_to_int,
                          dataset=dataset)
        print("###################################### Running Epoch: ", i, "########################################################")
        model = model.train()
        loss = train(train_loader, model, loss_fn, optimizer, test_loader)
        print("Epoch average Loss: ", loss)
        torch.save(model.state_dict(), "model_res_focal_loss_plt.pth")
        torch.save(optimizer.state_dict(), "opt_res_focal_loss_plt.pth")
        torch.save(scheduler.state_dict(), "scheduler_focal_loss_plt.pth")
        # if i%1 == 0:
        #     model = model.eval()
        #     mAP = validate(test_loader, 
        #                   model, 
        #                   loss_fn, 
        #                   inv_class=dataset.inv_class, 
        #                   direct_class=dataset.class_to_int,
        #                   dataset=dataset)

        scheduler.step()