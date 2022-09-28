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

from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import multiclass_nms, demo_postprocess, vis


# ----------------------------------------------------------------------------------------------------------------------
# 指定フォルダを検索し、なければ作成、あれば中身を空にする関数
# ----------------------------------------------------------------------------------------------------------------------
def mkfolder(target_folder):
    if os.path.isdir(target_folder):
        # cc = subprocess.call("rm -r " + target_folder + "/*", shell=True)
        print("folder exist !")
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

    # base_imgs = glob.glob(data_folder + "/00_base/*.jpg")
    # imgs_imgs = glob.glob(data_folder + "/10_image/*.jpg")
    #
    # result_list = []

    # for b in base_imgs:
    #     for t in imgs_imgs:
    #
    #         img0 = cv2.imread(b)
    #         img1 = cv2.imread(t)
    #
    #         vtr = imgsim.Vectorizer()
    #         vec0 = vtr.vectorize(img0)
    #         vec1 = vtr.vectorize(img1)
    #
    #         dist = imgsim.distance(vec0, vec1)
    #
    #         result_list.append([b, t, dist])
    #
    # df_out = pd.DataFrame(result_list)
    # df_out.to_csv(result_folder + "/result.csv", index=None, encoding="utf_8_sig")


# ------------------------------------------------------------- YOLOX test
#     # 画像の読み込み、前処理
#     input_shape = (416, 416)
#     origin_img = cv2.imread(data_folder + "/sample2.jpg")
#     img, ratio = preprocess(origin_img, input_shape)
#
#     # ONNXセッション
#     session = onnxruntime.InferenceSession(model_folder +"/yolox_nano_train.onnx")
#
#     # 推論＋後処理
#     ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
#     output = session.run(None, ort_inputs)
#     predictions = demo_postprocess(output[0], input_shape, p6=False)[0]
#
#     # xyxyへの変換＋NMS
#     boxes = predictions[:, :4]
#     scores = predictions[:, 4:5] * predictions[:, 5:]
#
#     boxes_xyxy = np.ones_like(boxes)
#     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
#     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
#     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
#     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
#     boxes_xyxy /= ratio
#     dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
#     if dets is not None:
#         final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
#         # BoundingBoxを描画する場合
#         inference_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
#                             conf=0.3, class_names=COCO_CLASSES)
#         cv2.imwrite(result_folder + "/output-image.jpg", inference_img)


# =============================================================================== YOLOX photos
    # 画像の読み込み、前処理
    input_shape = (416, 416)
    imgs_imgs = glob.glob(data_folder + "/88_test/*.jpg")

    # ONNXセッション
    session = onnxruntime.InferenceSession(model_folder +"/yolox_nano_train_20220928.onnx")

    for img in imgs_imgs:
        file_name = img[img.rfind("/")+1:img.rfind(".")]
        print(file_name)

        origin_img = cv2.imread(img)
        img, ratio = preprocess(origin_img, input_shape)

        # 推論＋後処理
        ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
        output = session.run(None, ort_inputs)
        predictions = demo_postprocess(output[0], input_shape, p6=False)[0]

        # xyxyへの変換＋NMS
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.01)
        print("---")

        if dets is not None:
            print("@@@@@")
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            # BoundingBoxを描画する場合
            inference_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                                conf=0.3, class_names=COCO_CLASSES)
        cv2.imwrite(result_folder + "/" + file_name + "_yolox.jpg", inference_img)
