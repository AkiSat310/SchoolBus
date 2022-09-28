# coding: utf-8
import glob
import sys
import os
import subprocess
# import cv2
import numpy as np
import pandas as pd
import imgsim
import cv2

import onnxruntime
from feat import Detector
import matplotlib.pyplot as plt
from PIL import Image


# rf: https://py-feat.org/content/intro.html#available-models
detector = Detector(
    face_model="RetinaFace",
    landmark_model="MobileFaceNet",
    au_model="JAANET",
    emotion_model="ResMaskNet"
)

# from feat.tests.utils import get_test_data_path


# ----------------------------------------------------------------------------------------------------------------------
# 指定フォルダを検索し、なければ作成、あれば中身を空にする関数
# ----------------------------------------------------------------------------------------------------------------------
def mkfolder(target_folder):
    if os.path.isdir(target_folder):
        print("--- folder exist!")
        # cc = subprocess.call("rm -r " + target_folder + "/*", shell=True)
    else:
        os.makedirs(target_folder)
    return target_folder

########################################################################################################################
########################################################################################################################
#　main関数
########################################################################################################################
########################################################################################################################
if __name__ == '__main__':

    #　前処理
    my_folder = os.getcwd()
    item_folder = my_folder + "/00_item"
    data_folder = my_folder + "/10_data"
    model_folder = my_folder + "/77_model"
    result_folder = mkfolder(my_folder + "/99_result")
    result_imgae_folder = mkfolder(my_folder + "/99_result/00_image")

    base_imgs = glob.glob(data_folder + "/00_base/*.jpg")
    test_imgs = glob.glob(data_folder + "/10_image/*.jpg")

    result_list = []

    test_image = data_folder + '/photo_0.jpg'
    # 画像の確認
    f, ax = plt.subplots()
    im = Image.open(test_image)
    # ax.imshow(im)
    # plt.show()


    image_prediction = detector.detect_image(test_image)
    # Show results
    image_prediction.plot_detections()
    print(image_prediction)
    df_img = pd.DataFrame(image_prediction)
    print(df_img)
    ax.imshow(im)
    plt.show()



