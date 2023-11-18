import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet as focal_loss
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
import operator

class Hierarchical_Task_Learning:
    def __init__(self,epoch0_loss,stat_epoch_nums=5):
        self.index2term = [*epoch0_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  #term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'seg_loss':[],
                           'size2d_loss':[], 
                           'offset2d_loss':[],
                           'offset3d_loss':['size2d_loss','offset2d_loss'], 
                           'size3d_loss':['size2d_loss','offset2d_loss'], 
                           'heading_loss':['size2d_loss','offset2d_loss'], 
                           'depth_loss':['size2d_loss','size3d_loss','offset2d_loss'],
                           'project_loss':['depth_loss'] 
                           }                                 
    def compute_weight(self,current_loss,epoch):
        T=140
        #compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        #update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
                self.init_diff[self.init_diff==0]=1
            c_weights = 1-(mean_diff/self.init_diff).relu().unsqueeze(0)
            
            time_value = min(((epoch-5)/(T-5)),1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]      
                    loss_weights[current_topic] = time_value**(1-control_weight)
            #pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)   
        return loss_weights
    def update_e0(self,eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)


class GupnetLoss(nn.Module):
    def __init__(self,epoch):
        super().__init__()
        self.stat = {}
        self.epoch = epoch


    def forward(self, preds, targets, task_uncertainties=None):

        seg_loss = self.compute_segmentation_loss(preds, targets)
        bbox2d_loss = self.compute_bbox2d_loss(preds, targets)
        bbox3d_loss = self.compute_bbox3d_loss(preds, targets)
        
        loss = seg_loss + bbox2d_loss + bbox3d_loss
        
        return loss, self.stat


    def compute_segmentation_loss(self, input, target):
        input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
        loss = focal_loss(input['heatmap'], target['heatmap'])
        self.stat['seg_loss'] = loss
        return loss


    def compute_bbox2d_loss(self, input, target):
        # compute size2d loss
        
        size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
        size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
        size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
        # compute offset2d loss
        offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
        offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
        offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')


        loss = offset2d_loss + size2d_loss   
        self.stat['offset2d_loss'] = offset2d_loss
        self.stat['size2d_loss'] = size2d_loss
        return loss


    def compute_bbox3d_loss(self, input, target, mask_type = 'mask_2d'):
        
        # compute depth loss        
        depth_input = input['depth'][input['train_tag']] 
        depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
        depth_target = extract_target_from_tensor(target['depth'], target[mask_type])
        depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)
        
        # compute offset3d loss
        offset3d_input = input['offset_3d'][input['train_tag']]  
        offset3d_target = extract_target_from_tensor(target['offset_3d'], target[mask_type])
        offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')
        
        # compute size3d loss
        size3d_input = input['size_3d'][input['train_tag']] 
        size3d_target = extract_target_from_tensor(target['size_3d'], target[mask_type])
        size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')*2/3+\
               laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])/3
        #size3d_loss = F.l1_loss(size3d_input[:,1:], size3d_target[:,1:], reduction='mean')+\
        #       laplacian_aleatoric_uncertainty_loss(size3d_input[:,0:1], size3d_target[:,0:1], input['h3d_log_variance'][input['train_tag']])
        # compute heading loss
        heading_loss = compute_heading_loss(input['heading'][input['train_tag']] ,
                                            target[mask_type],  ## NOTE
                                            target['heading_bin'],
                                            target['heading_res'])
        project_loss = compute_projectloss(input, target)
        # loss = depth_loss + offset3d_loss + size3d_loss + heading_loss
        loss = depth_loss + offset3d_loss + size3d_loss + heading_loss + 0.1*project_loss

        self.stat['project_loss'] = project_loss
        self.stat['depth_loss'] = depth_loss
        self.stat['offset3d_loss'] = offset3d_loss
        self.stat['size3d_loss'] = size3d_loss
        self.stat['heading_loss'] = heading_loss
        
        
        return loss




### ======================  auxiliary functions  =======================

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C


def extract_target_from_tensor(target, mask):
    return target[mask]

#compute heading loss two stage style  

def compute_heading_loss(input, mask, target_cls, target_reg):
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    target_cls = target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')
    
    # regression loss
    input_reg = input[:, 12:24]
    target_reg = target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss    

def compute_projectloss(input, target ):
    mask = target['mask_2d'].view(-1) #(B*K)
    cls_mean_size=torch.tensor([1.52563191462 ,1.62856739989, 3.88311640418],device='cuda')
    # size3d_input = input['size_3d'][input['train_tag']] 
    # real_size3d =size3d_input+cls_mean_size
    real_size3d = extract_target_from_tensor(target['size_3d'], target['mask_2d'])
    real_size3d=real_size3d+cls_mean_size #use real size 3d to  test
    # _, heading_res = input['heading'][input['train_tag']][:, 0:12],input['heading'][input['train_tag']][:, 12:24]#heading_input[:,0:12], heading_input[:,12:24] #bin好像可以不用
    depth_input =input['depth'][input['train_tag']][:, 0] #use in depth prediction

    # depth_target=extract_target_from_tensor(target['depth'], target['mask_2d'])#N*1
    # target_bin=target['heading_bin'].view(-1)
    # target_cls = target_bin.view(-1)
    # target_cls=target_bin[mask]
    # target_bin=target_bin[mask]

    pos_3d = extract_target_from_tensor(target['pos_3d'], target['mask_2d'])#N*3
    input_pos = torch.zeros_like(pos_3d)
    input_pos[:,:2] = pos_3d[:,:2]
    input_pos[:,2] = depth_input
    # trans_set = extract_target_from_tensor( target['trans'], target['mask_2d']) #(N,2,3)
    # trans_inv_set = extract_target_from_tensor( target['trans_inv_set'], target['mask_2d'])
    camera_set = extract_target_from_tensor(target['calib_set'],target['mask_2d'])
    # center_set = target['center_set'][:,:,0:2]#(4,50,2) ->4,50,2 ->(200)
    # center_x = center_set[:, :, 0:1].view(-1)[mask]* 4 #(200) -> object ,num 4 is downsample ratio and the result is real x center 
    # center_y = center_set[:,:, 1:2].view(-1)[mask]*4

    # cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    # input_reg = torch.sum(heading_res * cls_onehot, 1)
    # alpha =class2angle(target_cls,input_reg,to_label_format=True)

    # angle_per_class = 2 * torch.pi / 12
    # angle_center = target_cls * angle_per_class
    # angle = angle_center + input_reg
    # condition = angle > torch.pi
    # angle[condition] -= 2*torch.pi
    # r_y= alpha2ry(angle, center_x)
    r_y = extract_target_from_tensor(target['r_y'],target['mask_2d'])
    
    corner_3d_set = compute_3d_box_coordinate(real_size3d, input_pos, r_y)
    pts_2d = project_rect_to_image(corner_3d_set, camera_set) #(n,3,8)->(n,4)
    
    # affine_pts_2d = project_affine_transform(pts_2d, trans_set)
     
    target_project_box = target['project_2d_box'].view(-1,4)[mask] #->(200,4)->(n,4)
    
    loss_iou= iou_loss(pts_2d, target_project_box)

    return loss_iou



def iou_loss(predicted, target ):
    if (predicted.dim() == 1):
        predicted = predicted.unsqueeze(0)
    clamp_predicted = predicted.clone()
    clamp_predicted[:,0] = torch.clamp(predicted[:,0], 0, 1280)
    clamp_predicted[:,1] = torch.clamp(predicted[:,1], 0, 384)
    clamp_predicted[:,2] = torch.clamp(predicted[:,2], 0, 1280)
    clamp_predicted[:,3] = torch.clamp(predicted[:,3], 0, 384)
    x1 = torch.max(clamp_predicted[:, 0], target[:, 0])
    y1 = torch.max(clamp_predicted[:, 1], target[:, 1])
    x2 = torch.min(clamp_predicted[:, 2], target[:, 2])
    y2 = torch.min(clamp_predicted[:, 3], target[:, 3])
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0) #caculate the interseciton of area
    predicted_area = (clamp_predicted[:, 2] - clamp_predicted[:, 0]) * (clamp_predicted[:, 3] - clamp_predicted[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

    # Calculate the union area
    union_area = predicted_area + target_area - intersection_area
    iou = intersection_area / union_area

    # Compute IoU loss (1 - IoU)
    iou_loss = 1 - iou.mean()
    return iou_loss 

def project_affine_transform(pts_2d, trans ):
    """
     pts_2d : (N, 4)
     trans  : (N ,2,3)
    """
    new_pts_1 = torch.cat((pts_2d[:,:2],torch.ones(pts_2d.shape[0],device='cuda').unsqueeze(1)),dim=1).unsqueeze(2) #轉成齊次(N,2)->(N,3,1)
    new_pts_2 = torch.cat((pts_2d[:,2:],torch.ones(pts_2d.shape[0],device='cuda').unsqueeze(1)),dim=1).unsqueeze(2) #轉成齊次(N,2)->(N,3,1)
    new_pts_1 = trans@new_pts_1 #(N,2,3)@(N,3,1) ->(N,2,1)
    new_pts_2 = trans@new_pts_2 #(N,2,3)@(N,3,1) ->(N,2,1)
    return torch.cat((new_pts_1,new_pts_2),dim = 1).squeeze()#(N,2,1) cat (N,2,1) -> (N,4,1) ->(N,4)
    
def project_rect_to_image(corner_3d_set, camera_set):
    # camera=torch.tensor([7.215377e+02, 0, 6.095593e+02, 4.4857280e+01,
    #                     0, 7.2153770e+02, 1.728540e+02,2.1637910e-01,
    #                     0 , 0 , 1.00e+00, 2.7458840e-03], device='cuda').reshape(3,4).t().double()#要先reshape 再transpose 不能直接reshape 為4,3
    camera = camera_set.permute(0,2,1) #(N,3,4)
    
    pts_3d_rect = cart2hom(corner_3d_set.permute(0,2,1)) #(N,B,4)
    pts_2d = pts_3d_rect @ camera #(N, 8, 4) @ (N, 3,4) ->(N,B,3)
    # pts_2d =  camera @ pts_3d_rect
    # pts_x = pts_2d[:, :, 0] / pts_2d[:, :, 2]
    # pts_y = pts_2d[:, :, 1] / pts_2d[:, :, 2]
    pts_x = pts_2d[:, :, 0] / pts_3d_rect[:, :, 2] #除原本點的3d 深度
    pts_y = pts_2d[:, :, 1] / pts_3d_rect[:, :, 2]
    # pts_2d =torch.matmul(pts_3d_rect,camera.unsqueeze(0).repeat(len(corner_3d_set),1 ,1))
    # print(pts_2d)
    max_valuesx, _ = torch.max(pts_x, dim=1)#(n,8,1) -> n
    max_valuesy,_ = torch.max(pts_y, dim=1)
    min_valuex,_ = torch.min(pts_x, dim=1)
    min_valuey,_ = torch.min(pts_y, dim =1)
    return torch.stack((min_valuex, min_valuey, max_valuesx, max_valuesy),dim=1)#(n,4)

def cart2hom( pts_3d):
        ''' Input: b*nx3 points in Cartesian
            Oupput: b*nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[1]
        ones = torch.ones((pts_3d.shape[0],n,1), device='cuda') #(b*n*1)
        pts_3d_hom = torch.cat((pts_3d, ones), dim =2)#(b*n*3) cat (b*n*1) ->(b*n*4)
        return pts_3d_hom
def compute_3d_box_coordinate(real_size3d, pts_rect, r_y): #input is  regress h,w,l and gt x,y,z,yaw  
    """
    Return : 3xn in cam2 coordinate
    """
    dataset = torch.cat((real_size3d, pts_rect, r_y.view(-1,1)), dim = 1) #(n,3)cat(n,3)cat(n,1)->(n,7)
    object_num, c = real_size3d.size()

    data = dataset.permute(1, 0) # -> [7, N]
    h, w, l, x, y, z, yaw = data
    # R = torch.stack([
    #     torch.stack([
    #         torch.cos(yaw),
    #         torch.zeros_like(yaw),
    #         torch.sin(yaw)
    #     ], dim=1),
    #     torch.tensor([0, 1, 0], device='cuda')[None].repeat(object_num, 1),
    #     torch.stack([
    #         -torch.sin(yaw),
    #         torch.zeros_like(yaw),
    #         torch.cos(yaw),
    #     ], dim=1),
    # ], dim=2) #(n,3)stack(n,3)stack(n,3) -> (n,3,3)
    R = torch.stack([
        torch.stack([
            torch.cos(yaw),
            torch.zeros_like(yaw),
            -torch.sin(yaw)
        ], dim=1),
        torch.tensor([0, 1, 0], device='cuda')[None].repeat(object_num, 1),
        torch.stack([
            torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.cos(yaw),
        ], dim=1),
    ], dim=2) #(n,3)stack(n,3)stack(n,3) -> (n,3,3)
    
    # x_corners = torch.cat([
    #     (-l/2)[None].repeat(2, 1),
    #     (l/2)[None].repeat(4, 1),
    #     (-l/2)[None].repeat(2, 1)
    # ])
    x_corners = (l/2)[None].repeat(8, 1) #n->(1, n) ->(8,n)
    x_corners *= torch.tensor([1, 1, -1, -1, 1, 1, -1, -1], device='cuda')[:, None]
    y_corners = h[None].repeat(8, 1) #n->(1, n) ->(8,n)
    y_corners *= torch.tensor([0, 0, 0, 0, -1, -1, -1, -1], device='cuda')[:, None]
    # y_corners *= torch.tensor([1/2, 1/2, 1/2, 1/2, -1/2, -1/2, -1/2, -1/2], device='cuda')[:, None] #從中心點加的話
    z_corners = (w/2)[None].repeat(8, 1) #n->(1, n) ->(8,n)
    z_corners *= torch.tensor([1, -1, -1, 1, 1, -1, -1, 1], device='cuda')[:, None]

    corners = torch.stack([x_corners, y_corners, z_corners]).permute(2, 0, 1) #(3,8,n) -> (n,3,8)
    corners_3d_cam2 = R @ corners # [N, 3, 3] @ [N, 3, 8] -> [N, 3, 8]
    corners_3d_cam2 += torch.stack([x, y, z], dim=1).unsqueeze(2) #(n, 3)->(n, 3, 1)
    return corners_3d_cam2

def img_2_rect(u, v, depth):
    cu,cv,fu,fv= torch.tensor(6.4e+02, device='cuda'), torch.tensor(3.6e+02, device='cuda'), torch.tensor(1.418667e+03, device='cuda'), torch.tensor(1.418667e+03, device='cuda')
    
    # tx, ty = 0, 0 
    x = ((u - cu) * depth.view(-1)) / fu
    y = ((v - cv) * depth.view(-1)) / fv
    pts_rect= torch.cat((x.reshape(-1,1), y.reshape(-1,1), depth.reshape(-1,1)), dim=1)
    #pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth.reshape(-1, 1)), axis=1)
    return pts_rect

def alpha2ry(alpha,u):
    #x=u - 6.4e+02
    f= torch.full(u.size(),1.418667e+03,device='cuda')
    ry = alpha + torch.arctan2(u - 6.4e+02, f)
    
    return ry
'''    

def compute_heading_loss(input, ind, mask, target_cls, target_reg):
    """
    Args:
        input: features, shaped in B * C * H * W
        ind: positions in feature maps, shaped in B * 50
        mask: tags for valid samples, shaped in B * 50
        target_cls: cls anns, shaped in B * 50 * 1
        target_reg: reg anns, shaped in B * 50 * 1
    Returns:
    """
    input = _transpose_and_gather_feat(input, ind)   # B * C * H * W ---> B * K * C
    input = input.view(-1, 24)  # B * K * C  ---> (B*K) * C
    mask = mask.view(-1)   # B * K  ---> (B*K)
    target_cls = target_cls.view(-1)  # B * K * 1  ---> (B*K)
    target_reg = target_reg.view(-1)  # B * K * 1  ---> (B*K)

    # classification loss
    input_cls = input[:, 0:12]
    input_cls, target_cls = input_cls[mask], target_cls[mask]
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    # regression loss
    input_reg = input[:, 12:24]
    input_reg, target_reg = input_reg[mask], target_reg[mask]
    cls_onehot = torch.zeros(target_cls.shape[0], 12).cuda().scatter_(dim=1, index=target_cls.view(-1, 1), value=1)
    input_reg = torch.sum(input_reg * cls_onehot, 1)
    reg_loss = F.l1_loss(input_reg, target_reg, reduction='mean')
    
    return cls_loss + reg_loss
'''


if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

