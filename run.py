import iftv


if __name__ == "__main__":
    # params list
    binLen = 2000
    treeNum = 256
    treeSampleNum = 256
    alpha = 0.25
    threshold = 0.01
    for i in range(10,30):
        num = i + 1
        chrName = "ref.fa" 
        rdName = "sim_10x_" + str(num) + ".sort.bam"
        print(chrName)
        print(rdName)
        chrFile = "/media/anabas/C0EAA201EAA1F3B6/Art_simu/sim_0.7/" + chrName
        #chrName = chrFile.split('/')[7]
        rdFile = "/media/anabas/C0EAA201EAA1F3B6/Art_simu/sim_0.7/" + rdName
        outputFile = "sim_10x_change_output" + str(num) + ".txt" 
        statisticFile = "statistic_10x_change.txt"
        calculateFile = "calculateFile_abCN.txt"
        beforeFile = "beforefile_CN.txt"

        params = (binLen, chrFile, rdFile, outputFile, statisticFile, treeNum, treeSampleNum, alpha, threshold, calculateFile , beforeFile)
        iftv.main(params)
