from findoverLap import findOverlap
import cv2
from PIL import Image
import handleBright
from BlendMod import blendProcess
from time import time
from handleColor import  coltmp_tranfer

def handleOverLop(human, scenes):
    # 取得所有场景的mask和Lap
    masks = []
    laps = []
    for scene in scenes:
        lap, mask = findOverlap(human, scene)
        masks.append(mask)
        laps.append(lap)
    # 将他们全部保存
    for i in range(0, len(masks)):
        cv2.imwrite(address + maskNames[i], masks[i])
    for i in range(0, len(laps)):
        cv2.imwrite(address + lapNames[i], laps[i])


def handleBrightProcess(human, scenes, laps, masks):
    pass


if __name__ == "__main__":
    # 使用姓名数组便于打开所有文件，并且有益于扩展
    start = time()
    address = "./image/"
    srcNames = ["left.jpg", "right.jpg", "top.jpg", "leftTop.jpg", "rightTop.jpg"]
    lapNames = ["leftLap.jpg", "rightLap.jpg", "topLap.jpg", "leftTopLap.jpg", "rightTopLap.jpg"]
    maskNames = ["leftMask.jpg", "rightMask.jpg", "topMask.jpg", "leftTopMask.jpg", "rightTopMask.jpg"]
    # 用opencv的方法打开文件来使用图片对齐操作
    scenes = []
    for srcName in srcNames:
        scenes.append(cv2.imread(address + srcName))
    human = cv2.imread(address + "human.jpg")

    handleOverLop(human, scenes)
    print("图像匹配已完成")

    # 为了兼容性，重新用PIL打开所有文件
    scenes, laps, masks = [], [], []
    for srcName in srcNames:
        scenes.append(Image.open(address + srcName))
    for lapName in lapNames:
        laps.append(Image.open(address + lapName))
    for maskName in maskNames:
        masks.append(Image.open(address + maskName))

    human = Image.open(address + "human.jpg")
    new_scenes = []
    for i in range(0,len(scenes)):
        new_scenes.append(coltmp_tranfer(scenes[i]))
        new_scenes[i].save(address + "col" + srcNames[i])
    for i in range(0, len(srcNames)):
        new_scenes[i]=handleBright.weighted_bright_tranfer(human, new_scenes[i], masks[i],laps[i])
    for i in range(0, len(srcNames)):
        new_scenes[i].save(address + "new" + srcNames[i])
    human = coltmp_tranfer(human)
    print("图像亮度调节和色温调节已完成")

    img = []
    img.append(cv2.imread(address + 'human.jpg'))
    img.append(cv2.imread(address + 'newtop.jpg'))
    img.append(cv2.imread(address + 'newright.jpg'))
    img.append(cv2.imread(address + 'newrightTop.jpg'))
    img.append(cv2.imread(address + 'newleft.jpg'))
    img.append(cv2.imread(address + 'newleftTop.jpg'))
    blendProcess(img)
    print("图像拼接已完成，请查看result.jpg文件")
    end = time()
    print("程序运行了"+str(end-start)+"秒")