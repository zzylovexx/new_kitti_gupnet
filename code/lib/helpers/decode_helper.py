import numpy as np
import torch
import torch.nn as nn
from lib.datasets.utils import class2angle

def decode_detections(dets, info, calibs, cls_mean_size, threshold):
    '''
    NOTE: THIS IS A NUMPY FUNCTION
    input: dets, numpy array, shape in [batch x max_dets x dim]
    input: img_info, dict, necessary information of input images
    input: calibs, corresponding calibs for the input batch
    output:
    '''
    results = {}
    for i in range(dets.shape[0]):  # batch
        preds = []
        for j in range(dets.shape[1]):  # max_dets
            cls_id = int(dets[i, j, 0])
            score = dets[i, j, 1]
            if score < threshold: continue
            
            # 2d bboxs decoding
            x = dets[i, j, 2] * info['bbox_downsample_ratio'][i][0]
            y = dets[i, j, 3] * info['bbox_downsample_ratio'][i][1]
            w = dets[i, j, 4] * info['bbox_downsample_ratio'][i][0]
            h = dets[i, j, 5] * info['bbox_downsample_ratio'][i][1]
            bbox = [x-w/2, y-h/2, x+w/2, y+h/2]

            # 3d bboxs decoding
            # depth decoding
            depth = dets[i, j, 6]

            # heading angle decoding
            alpha = get_heading_angle(dets[i, j, 7:31])
            ry = calibs[i].alpha2ry(alpha, x)

            # dimensions decoding
            dimensions = dets[i, j, 31:34]
            dimensions += cls_mean_size[int(cls_id)]
            if True in (dimensions<0.0): continue

            # positions decoding
            x3d = dets[i, j, 34] * info['bbox_downsample_ratio'][i][0]
            y3d = dets[i, j, 35] * info['bbox_downsample_ratio'][i][1]
            locations = calibs[i].img_to_rect(x3d, y3d, depth).reshape(-1)
            locations[1] += dimensions[0] / 2

            preds.append([cls_id, alpha] + bbox + dimensions.tolist() + locations.tolist() + [ry, score])
        results[info['img_id'][i]] = preds
    return results
    
#two stage style    
def extract_dets_from_outputs(outputs,targets,use_2dgt, K=50):
    # get src outputs
    heatmap = outputs['heatmap']
    size_2d = outputs['size_2d']
    offset_2d = outputs['offset_2d']    
    
    batch, channel, height, width = heatmap.size() # get shape

    if use_2dgt: #用2d gt的話要創建一些map用以儲存相關data
        mask=targets["mask_2d"]
        indices=targets["indices"]
        zero_depth_map = torch.zeros(batch,
                                    height * width,
                                    1,
                                    device='cuda')  # 創見(4,56320,1) map 其值都為0
        
        zero_3Dsize_map = torch.zeros(batch, height * width, 3,device='cuda')  # 創見(4,56320,1) map 其值都為0
        zero_heading_map = torch.zeros(batch, height * width, 24, device='cuda')
        zero_offset3d_map = torch.zeros(batch, height * width, 2, device='cuda')


    heading = outputs['heading'].view(batch,K,-1)
    depth = outputs['depth'].view(batch,K,-1)[:,:,0:1]
    size_3d = outputs['size_3d'].view(batch,K,-1)
    offset_3d = outputs['offset_3d'].view(batch,K,-1)
    # heatmap= torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)

    if use_2dgt: #這邊主要是因為經過nms 之後inds會跟indices 不一樣 ,因此需要作出相對應的map,我的方法是將其還原成(b,h*w,c)的map再用新的inds去對應
        zero_depth_map.scatter_(1, indices.unsqueeze(2).expand( batch, K, 1), depth)
        zero_depth_map= zero_depth_map.view(batch,1,height,width)#(B,C,H,W) 

        zero_3Dsize_map.scatter_(1, indices.unsqueeze(2).expand( batch, K, 3), size_3d)
        zero_3Dsize_map= zero_3Dsize_map.view(batch,height,width,3).permute(0,3,1,2).contiguous()#(B,C,H,W) 一定要先view 再permute 
   
        zero_heading_map.scatter_(1, indices.unsqueeze(2).expand( batch, K, 24), heading)
        zero_heading_map= zero_heading_map.view( batch, height, width, 24).permute(0, 3, 1, 2).contiguous()#(B,C,H,W)

        zero_offset3d_map.scatter_(1, indices.unsqueeze(2).expand( batch, K, 2), offset_3d)
        zero_offset3d_map= zero_offset3d_map.view(batch, height, width, 2).permute(0, 3, 1, 2).contiguous()#(B,C,H,W)
    else:
        heatmap= torch.clamp(heatmap.sigmoid_(), min=1e-4, max=1 - 1e-4)
        heatmap = _nms(heatmap)
        # perform nms on heatmaps
    scores, inds, cls_ids, xs, ys = _topk(heatmap, K=K)
    
    offset_2d = _transpose_and_gather_feat(offset_2d, inds)
    offset_2d = offset_2d.view(batch, K, 2)
    xs2d = xs.view(batch, K, 1) + offset_2d[:, :, 0:1]
    ys2d = ys.view(batch, K, 1) + offset_2d[:, :, 1:2]
  
    if use_2dgt:
        new_depth= _transpose_and_gather_feat(zero_depth_map,inds) #新的map 透過由nms 抽出來的inds來提取所需要的value
        new_3d_size = _transpose_and_gather_feat(zero_3Dsize_map,inds)
        new_heading= _transpose_and_gather_feat(zero_heading_map,inds)
        new_3d_offset=_transpose_and_gather_feat(zero_offset3d_map,inds)
        xs3d = xs.view(batch, K, 1) + new_3d_offset[:, :, 0:1] #this is not very 確定 ,這是由2d gt 
        ys3d = ys.view(batch, K, 1) + new_3d_offset[:, :, 1:2]
    elif use_2dgt ==False:
        xs3d = xs.view(batch, K, 1) + offset_3d[:, :, 0:1]
        ys3d = ys.view(batch, K, 1) + offset_3d[:, :, 1:2]

    cls_ids = cls_ids.view(batch, K, 1).float()
    if use_2dgt:
        scores = scores.view(batch, K, 1)
    else:
        depth_score = (-(0.5*outputs['depth'].view(batch,K,-1)[:,:,1:2]).exp()).exp()
        scores = scores.view(batch, K, 1)*depth_score
    # depth_score = (-(0.5*outputs['depth'].view(batch,K,-1)[:,:,1:2]).exp()).exp()
    # scores = scores.view(batch, K, 1)*depth_score

    # check shape
    xs2d = xs2d.view(batch, K, 1)
    ys2d = ys2d.view(batch, K, 1)
    xs3d = xs3d.view(batch, K, 1)
    ys3d = ys3d.view(batch, K, 1)

    size_2d = _transpose_and_gather_feat(size_2d, inds)
    size_2d = size_2d.view(batch, K, 2)

    if use_2dgt: 
        scores[scores!=1]=0 #不為1 都設為0 用2D GT 的才要用這種形式
        detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, new_depth, new_heading, new_3d_size, xs3d, ys3d], dim=2)
    else:
        detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d], dim=2)
    # detections = torch.cat([cls_ids, scores, xs2d, ys2d, size_2d, depth, heading, size_3d, xs3d, ys3d], dim=2)

    return detections


############### auxiliary function ############


def _nms(heatmap, kernel=3):
    padding = (kernel - 1) // 2
    heatmapmax = nn.functional.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=padding) #(3,3) kernel 內最大的的值
    keep = (heatmapmax == heatmap).float() #
    return heatmap * keep


def _topk(heatmap, K=50):
    batch, cat, height, width = heatmap.size()

    # batch * cls_ids * 50
    topk_scores, topk_inds = torch.topk(heatmap.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # batch * cls_ids * 50
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_cls_ids = (topk_ind / K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_cls_ids, topk_xs, topk_ys


def _gather_feat(feat, ind, mask=None):
    '''
    Args:
        feat: tensor shaped in B * (H*W) * C
        ind:  tensor shaped in B * K (default: 50)
        mask: tensor shaped in B * K (default: 50)

    Returns: tensor shaped in B * K or B * sum(mask)
    '''
    dim  = feat.size(2)  # get channel dim
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # B*len(ind) --> B*len(ind)*1 --> B*len(ind)*C
    feat = feat.gather(1, ind)  # B*(HW)*C ---> B*K*C
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # B*50 ---> B*K*1 --> B*K*C
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    '''
    Args:
        feat: feature maps shaped in B * C * H * W
        ind: indices tensor shaped in B * K
    Returns:
    '''
    feat = feat.permute(0, 2, 3, 1).contiguous()   # B * C * H * W ---> B * H * W * C
    feat = feat.view(feat.size(0), -1, feat.size(3))   # B * H * W * C ---> B * (H*W) * C
    feat = _gather_feat(feat, ind)     # B * len(ind) * C
    return feat


def get_heading_angle(heading):
    heading_bin, heading_res = heading[0:12], heading[12:24]
    cls = np.argmax(heading_bin)
    res = heading_res[cls]
    return class2angle(cls, res, to_label_format=True)



if __name__ == '__main__':
    ## testing
    from lib.datasets.kitti import KITTI
    from torch.utils.data import DataLoader

    dataset = KITTI('../../data', 'train')
    dataloader = DataLoader(dataset=dataset, batch_size=2)
