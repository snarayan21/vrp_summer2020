import cv2
import numpy as np 
from matplotlib import pyplot as plt
import shapely.geometry as geom

img = cv2.imread('/Users/saaketh/Desktop/vrp_summer2020/edited_road_map_w_bkg.png',0)
ret,thresh = cv2.threshold(img,5,230, cv2.THRESH_BINARY)
height, width = img.shape
print("height and width : ",height, width)
size = img.size
#imgplot = plt.imshow(thresh, 'gray')
x = []
y = []
for i in range(len(thresh)):
    found = False
    ys = []
    for j in range(len(thresh)):
        curr = thresh[j,i]
        if (curr == 0 and found == False):
            found = True
            ys.append(len(thresh) - (j+1))
        elif (curr == 0 and found == True):
            ys.append(len(thresh) - (j+1))
        elif (curr != 0 and found == True):
            y.append(sum(ys)/len(ys))
            x.append(i)
            break
        else:
            pass


""" while (len(x) > 1000):
    i = np.random.randint(0,len(x)-1)
    x.pop(i)
    y.pop(i)

print(len(x))
"""
line = geom.LineString(list(zip(x,y)))
xLine, yLine = line.xy
point = geom.Point(100, 100)
print(point.distance(line))
point_on_line = line.interpolate(line.project(point))
num_points = 9  # includes first and last
#new_points = [line.interpolate(i/float(num_points - 1), normalized=True) for i in range(num_points)]
ip1 = line.interpolate(0.7, normalized=True)
ip2 = line.interpolate(1, normalized=True)
dist1 = line.project(ip1)
dist2 = line.project(ip2)
diff = abs(dist1-dist2)/(num_points-1)
new_points = [line.interpolate(dist1 + (i)*(diff)) for i in range(num_points)]

dpi = 192
plt.figure(figsize=(1000/dpi, 1000/dpi))
plt.plot(x, y, 'ro', ms=1)
plt.plot(xLine,yLine,'b', lw=1)
plt.plot([point.x, point_on_line.x],[point.y, point_on_line.y],'go-', lw=2)
plt.plot(ip1.x, ip1.y, 'yo', ms=3)
plt.plot(ip2.x, ip2.y, 'yo', ms=3)
for i in range(num_points):
    plt.plot(new_points[i].x, new_points[i].y, 'co', ms=3)
plt.xlim([0,len(thresh)])
plt.ylim([0,len(thresh)])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()