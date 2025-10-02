import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms.functional import resize

from mm_sam.models.sam import SAMbyWMMF
from mm_sam.datasets.neu_rssdds_aug import NEUDataset
from mm_sam.train_agents.sam import BaseSAMTrainAgent

def _build_optimizer(model, optim_name, lr, betas, weight_decay, layer_decay=None):
    if optim_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optim_name}")

class NEUTrainAgent(BaseSAMTrainAgent):
    def agent_init(self, **kwargs):
        self.sam: SAMbyWMMF = SAMbyWMMF(
            model_type="vit_b",
            x_data_field="depth_images",
            x_channel_num=3,
            x_norm_type="min-max",
            x_lora_rank=kwargs.get('x_lora_rank', 4),
        )

        self.optimizer = _build_optimizer(
            model=self.sam,
            optim_name="AdamW",
            lr=kwargs.get('lr', 1e-4),
            betas=(0.9, 0.999),
            weight_decay=0.1,
            layer_decay=0.8
        )
        self.criterion = torch.nn.BCEWithLogitsLoss()

        train_bs = kwargs['train_bs']
        train_workers_per_bs = int(train_bs * kwargs['train_workers_per_bs'])
        test_bs = kwargs['test_bs']
        test_workers_per_bs = int(test_bs * kwargs['test_workers_per_bs'])

        self.train_dataset = NEUDataset(is_train=True, data_dir='../datasets/NEU-RSDDS-AUG', used_prompts=['box', 'point'], transforms=kwargs.get('train_transforms'))
        self.test_dataset = NEUDataset(is_train=False, data_dir='../datasets/NEU-RSDDS-AUG', transforms=kwargs.get('test_transforms'))

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True, num_workers=train_workers_per_bs, collate_fn=NEUDataset.collate_fn)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=test_bs, shuffle=False, num_workers=test_workers_per_bs, collate_fn=NEUDataset.collate_fn)

        self.output_dir = '/hy-tmp/output/predictions/'
        os.makedirs(self.output_dir, exist_ok=True)

        self.after_agent_init()

    def train_step(self, batch, epoch, step):
        ori_img_size = batch['original_size']

        processed_rgb, processed_points, processed_boxes = self.sam.preprocess(
            imgs=batch['rgb_images'],
            point_coords=batch['point_coords'],
            box_coords=batch['box_coords'],
            ori_img_size=ori_img_size
        )

        x_images_processed, _, _ = self.sam.preprocess(imgs=batch['depth_images'], ori_img_size=ori_img_size)

        rgb_feats = self.sam.image_encoder(processed_rgb)
        x_feats = self.sam.x_encoder(x_images_processed)

        fused_feats, _ = self.sam.fusion_module(feat_list=[rgb_feats, x_feats])

        points, boxes, masks = self.sam.convert_raw_prompts_to_triple(
            point_coords=processed_points,
            point_labels=batch['point_labels'],
            box_coords=processed_boxes,
            noisy_masks=batch['noisy_object_masks'],
            batch_size=len(fused_feats)
        )

        outputs = []
        for batch_idx in range(len(fused_feats)):
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=points[batch_idx], boxes=boxes[batch_idx], masks=masks[batch_idx],
            )

            low_res_masks, _ = self.sam.mask_decoder(
                image_embeddings=fused_feats[batch_idx].unsqueeze(0),
                prompt_encoder=self.sam.prompt_encoder,
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                multimask_output=False,
            )

            ori_res_masks = self.sam.postprocess(
                pred_masks=low_res_masks,
                output_mask_size=ori_img_size[batch_idx]
            )
            outputs.append(ori_res_masks)

        pred_masks = torch.cat(outputs)
        gt_masks = torch.stack(batch['gt_masks'], dim=0).to(dtype=torch.float32, device=self.device)

        loss = self.criterion(pred_masks, gt_masks.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss}

    def test_step(self, batch, iter_name=None):
        for i in range(len(batch['index_name'])):
            single_sample = {k: [v[i]] for k, v in batch.items()}

            self.sam.set_infer_img(data_dict=single_sample)

            pred_masks, _ = self.sam.infer(
                box_coords=single_sample['box_coords'],
                output_mask_size=[single_sample['original_size'][0]]
            )

            mask = pred_masks[0]
            index_name = single_sample['index_name'][0]

            mask_np = mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            img = Image.fromarray(mask_np, 'L')

            output_path = os.path.join(self.output_dir, f"{index_name}.png")
            img.save(output_path)

    def get_test_results(self):
        return {}

    def state_dict(self):
        return {'model': self.sam.state_dict(), 'optimizer': self.optimizer.state_dict()}

    def load_model_state_dict(self, state_dict):
        if 'model' in state_dict:
            self.sam.load_state_dict(state_dict['model'])
        else:
            self.sam.load_state_dict(state_dict)

    def optim_step(self, loss_dict, **kwargs):
        pass

    def get_optim_lr(self):
        return {"main_opt": self.optimizer.param_groups[0]['lr']}

    def valid_step(self, batch, iter_name=None):
        pass

    def get_valid_results(self):
        return {}