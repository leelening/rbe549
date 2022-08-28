"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import kornia

# Don't generate pyc codes
sys.dont_write_bytecode = True


def photometric_loss(delta, img_a, patch_b, corners):
    corners_hat = corners + delta

    # in order to apply transform and center crop,
    # subtract points by top-left corner (corners[N, 0])
    corners = corners - corners[:, 0].view(-1, 1, 2)

    h = kornia.get_perspective_transform(corners, corners_hat)

    h_inv = torch.inverse(h)
    patch_b_hat = kornia.warp_perspective(img_a, h_inv, (128, 128))

    return F.l1_loss(patch_b_hat, patch_b)


class HomographyModel(nn.Module):
    def __init__(self, InputSize, OutputSize):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################

    def forward(self, xb):
        """
        Input:
        xb is a MiniBatch of the current image
        Outputs:
        out - output of the network
        """

        #############################
        # Fill your network structure of choice here!
        #############################

        return out
