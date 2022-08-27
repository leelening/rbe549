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
        #############################
        # Fill your network initialization of choice here!
        #############################

    def forward(self, img):
        """
        The forward function

        :param img: input image
        :type img: torch.sensor
        :return: the output
        :rtype: torch.sensor
        """
        #############################
        # Fill your network here!
        #############################
        return output
