import matplotlib.pyplot as plt

def main():
    """ print("plotting figures based on data.")
    imagename = "numvehiclescomparison_vt-vd_ratio_1to1_routes_7.png"
    title = "Makespan vs. #of drones (vT/vD = 1, 7 routes)"
    y = [487, 327, 286, 252, 232, 213, 204, 204]
    x = list(range(1,len(y)+1))
    dpi = 192
    plt.figure(figsize=(1000/dpi, 1000/dpi))
    plt.plot(x,y,'ko-',markersize=7)
    plt.title(title)
    plt.xlabel('# of drones')
    plt.ylabel('makespan')
    plt.xlim(0,max(x)+1)
    plt.ylim(0, max(y) * 1.2)
    plt.savefig(imagename, dpi=dpi)
    plt.show() """

    print("plotting figures based on data.")
    imagename = "velocityratiocomparison_3drones_routes_7.png"
    title = "Makespan vs. vT/vD ratio (3 drones, 7 routes)"
    y = [8706, 3129, 1130, 552, 295, 208, 174, 173, 166]
    truckvels = [2.78255940e-01, 7.74263683e-01, 2.15443469e+00, 5.99484250e+00, 1.66810054e+01, 4.64158883e+01, 1.29154967e+02, 3.59381366e+02, 1.00000000e+03]
    dronevel = 15.0
    ratiovels = []
    for i in range(len(truckvels)):
        ratiovels.append(truckvels[i]/dronevel)

    dpi = 192
    plt.figure(figsize=(1000/dpi, 1000/dpi))
    plt.plot(ratiovels,y,'ko-',markersize=7)
    plt.title(title)
    plt.xlabel('truck velocity/drone velocity ratio')
    plt.ylabel('makespan')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(min(ratiovels)/2,max(ratiovels)*2)
    plt.ylim(1, max(y) * 2)
    plt.savefig(imagename, dpi=dpi)
    plt.show()


if __name__ == '__main__':
    main()