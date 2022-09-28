import os, sys
import cv2
import glob
import numpy as np
from matplotlib import pyplot as plt

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
    result_folder = mkfolder(my_folder + "/99_result/20_template")


    test_imgs = glob.glob(data_folder + "/88_test/*.png")

    # for t in test_imgs:
    #     file_name = t[t.rfind("/")+1:t.rfind(".")]
    #     print(file_name)
    #
    #     img_rgb = cv2.imread(t)
    #     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #
    #     template = cv2.imread(data_folder + '/99_teach/teach2.png')
    #     # template = cv2.imread(data_folder + '/99_teach/teach.jpeg')
    #     template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    #
    #     res = cv2.matchTemplate(img_gray,template_gray,cv2.TM_CCOEFF_NORMED)
    #
    #     threshold = 0.3
    #     loc = np.where( res >= threshold)
    #
    #     h, w = template_gray.shape
    #
    #     for pt in zip(*loc[::-1]):
    #         cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    #
    #     cv2.imwrite(result_folder + "/" + file_name + "_tmp.jpg",img_rgb)

    # 対象画像を指定
    base_image_path = '＜file path＞/＜file name＞'
    temp_image_path = '＜file path＞/＜file name＞'
    template = cv2.imread(data_folder + '/99_teach/teach.jpeg')


    # 画像をグレースケールで読み込み
    gray_temp_src = cv2.imread(data_folder + '/99_teach/teach2.png', 0)
    # gray_temp_src = cv2.imread(data_folder + '/99_teach/teach.jpeg', 0)

    for t in test_imgs:
        file_name = t[t.rfind("/")+1:t.rfind(".")]
        print(file_name)

        base_src = cv2.imread(t)
        gray_base_src = cv2.cvtColor(base_src, cv2.COLOR_BGR2GRAY)

        # マッチング結果書き出し準備
        # 画像をBGRカラーで読み込み
        color_base_src = cv2.imread(t, 1)
        color_temp_src = cv2.imread(data_folder + '/99_teach/teach2.png', 1)
        # color_temp_src = cv2.imread(data_folder + '/99_teach/teach.jpeg', 1)

        # 特徴点の検出
        type = cv2.AKAZE_create()
        kp_01, des_01 = type.detectAndCompute(gray_base_src, None)
        kp_02, des_02 = type.detectAndCompute(gray_temp_src, None)

        # マッチング処理
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.match(des_01, des_02)
        matches = sorted(matches, key=lambda x: x.distance)
        mutch_image_src = cv2.drawMatches(color_base_src, kp_01, color_temp_src, kp_02, matches[:10], None, flags=2)

        # 結果の表示
        # cv2.imshow("mutch_image_src", mutch_image_src)
        # cv2.imshow("02_result08", mutch_image_src)
        #
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite(result_folder + "/" + file_name + "_tmp.jpg",mutch_image_src)



