#
# Copyright (C) 2022 Apple Inc. All rights reserved.
#

'''
Render test views, and report the metrics
'''

import os
import argparse

import cv2
import imageio
import torch
import numpy as np
import skimage
from skimage.metrics import structural_similarity as ssim
import lpips

loss_fn_alex = lpips.LPIPS(net='alex')

from models import human_nerf
from utils import render_utils, utils
from data_io import neuman_helper
from options import options


def eval_metrics(gts, preds):
    results = {
        'ssim': [],
        'psnr': [],
        'lpips': []
    }
    for gt, pred in zip(gts, preds):
        results['ssim'].append(ssim(pred, gt, multichannel=True))
        results['psnr'].append(skimage.metrics.peak_signal_noise_ratio(gt, pred))
        results['lpips'].append(
            float(loss_fn_alex(utils.np_img_to_torch_img(pred[None])/127.5-1, utils.np_img_to_torch_img(gt[None])/127.5-1)[0, 0, 0, 0].data)
        )
    for k, v in results.items():
        results[k] = np.mean(v)
    return results


def optimize_pose_with_nerf(opt, cap, net, iters=1000, save_every=10):
    # TODO: optimize the pose with the trained NeRF
    pass


def main(opt):
    train_split, _, test_split = neuman_helper.create_split_files(opt.scene_dir)
    test_split = train_split
    test_views = neuman_helper.read_text(test_split)
    scene = neuman_helper.NeuManReader.read_scene(
        opt.scene_dir,
        tgt_size=opt.render_size,
        normalize=opt.normalize,
        bkg_range_scale=opt.bkg_range_scale,
        human_range_scale=opt.human_range_scale,
        smpl_type='optimized'
    )
    if opt.geo_threshold < 0:
        bones = []
        for i in range(len(scene.captures)):
            bones.append(np.linalg.norm(scene.smpls[i]['joints_3d'][3] - scene.smpls[i]['joints_3d'][0]))
        opt.geo_threshold = np.mean(bones)
    net = human_nerf.HumanNeRF(opt)
    weights = torch.load(opt.weights_path, map_location='cpu')
    utils.safe_load_weights(net, weights['hybrid_model_state_dict'])

    preds = []
    gts = []
    for view_name in test_views:
        cap = scene[view_name]
        i = cap.frame_id['frame_id']
        out, out_depth, out_normal = render_utils.render_hybrid_nerf(
            net,
            cap,
            scene.verts[i],
            scene.faces,
            scene.Ts[i],
            rays_per_batch=opt.rays_per_batch,
            samples_per_ray=opt.samples_per_ray,
            geo_threshold=opt.geo_threshold,
            return_depth=True,
        )
        save_path = os.path.join('./demo', f'train_views/{os.path.basename(opt.scene_dir)}', f'out_{str(i).zfill(4)}.png')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        out = (out * 255).astype(np.uint8)
        imageio.imsave(save_path, out)

        np.save(save_path.replace(".png", "depth.npy"), out_depth)
        depth_map = out_depth
        depth_map = depth_map  / depth_map.max()
        depth_map_gray = (depth_map * 255).astype(np.uint8)
        cv2.imwrite(save_path.replace(".png", "depth.png"), depth_map_gray)
        depth_map_color = cv2.applyColorMap(depth_map_gray, cv2.COLORMAP_JET)
        cv2.imwrite(save_path.replace(".png", "depth_color.png"), depth_map_color)

        np.save(save_path.replace(".png", "normal.npy"), out_normal)
        normal_map = (out_normal + 1) * 127.5
        cv2.imwrite(save_path.replace(".png", "normal.png"), normal_map)

        preds.append(imageio.imread(save_path))
        gts.append(cap.image)
    print(eval_metrics(gts, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    options.set_general_option(parser)
    opt, _ = parser.parse_known_args()

    options.set_nerf_option(parser)
    options.set_pe_option(parser)
    options.set_render_option(parser)
    options.set_trajectory_option(parser)
    parser.add_argument('--scene_dir', required=True, type=str, help='scene directory')
    parser.add_argument('--image_dir', required=False, type=str, default=None, help='image directory')
    parser.add_argument('--out_dir', default='./out', type=str, help='weights dir')
    parser.add_argument('--offset_scale', default=1.0, type=float, help='scale the predicted offset')
    parser.add_argument('--geo_threshold', default=-1, type=float, help='')
    parser.add_argument('--normalize', default=True, type=options.str2bool, help='')
    parser.add_argument('--bkg_range_scale', default=3, type=float, help='extend near/far range for background')
    parser.add_argument('--human_range_scale', default=1.5, type=float, help='extend near/far range for human')
    parser.add_argument('--num_offset_nets', default=1, type=int, help='how many offset networks')
    parser.add_argument('--offset_scale_type', default='linear', type=str, help='no/linear/tanh')

    opt = parser.parse_args()
    assert opt.geo_threshold == -1, 'please use auto geo_threshold'
    if opt.render_h is None:
        opt.render_size = None
    else:
        opt.render_size = (opt.render_h, opt.render_w)
    main(opt)
