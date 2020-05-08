import random
import os
import sys
def genRandomWorld(fileName, cube_number, W=20, H=20):
    fr = open(fileName, 'w')
    path = os.path.split(os.path.realpath(__file__))[0]
    headfile = path + "/head.txt"
    tailfile = path + "/tail.txt"
    cubefile = path + "/cube.txt"
    hf = open(headfile, "r")
    locationList = random.sample(range(0, W*H), cube_number+2)
    for line in hf.readlines():
        fr.write(line)
    hf.close()
    cf = open(cubefile, "r")
    lines = cf.readlines()
    for i in range(cube_number):
        x = locationList[i] / W
        y = locationList[i] % W
        writeCube(fr, lines, i, x, y)
    cf.close()
    tf = open(tailfile,"r")
    for line in tf.readlines():
        fr.write(line)
    fr.close()
    return locationList[cube_number] / W, locationList[cube_number] % W, \
           locationList[cube_number+1] / W, locationList[cube_number+1] % W,
def writeCube(outFile,cubeFileLines,id,x,y):
    global W, H
    outFile.write("    <model name='unit_box_"+str(id)+"'>\n")
    outFile.write("      <pose>" + str(x) + " " + str(y) + " 0 0 0 0</pose>\n")
    for line in cubeFileLines:
        outFile.write(line)
    return
def test():
    genRandomWorld("../worlds/random.world", 20)
if __name__ == "__main__":
    test()