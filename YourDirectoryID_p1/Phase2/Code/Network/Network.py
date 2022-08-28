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
import numpy as np

# Don't generate pyc codes
sys.dont_write_bytecode = True


class HomographyModel(nn.Model):
    def __init__(self, input_size, output_size):
        """
        Initialization

        :param input_size: the size of the input
        :type input_size: int
        :param output_size: the size of the output
        :type output_size: int
        """
        super().__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        #############################
        # Fill your network initialization of choice here!
        #############################

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, input_img):
        """
        The forward function

        :param input_img: input image
        :type input_img: torch.sensor
        :return: the output of the network
        :rtype: torch.sensor
        """
        #############################
        # Fill your network here!
        #############################
        return output
