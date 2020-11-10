import sys
fin = open(sys.argv[1]).readlines()
ppls = [100]
r2s = [0]
for line in fin:
    if line.startswith("validation"):
        itr = line.strip().split(",")[0].split()[-1]
        ppl = float(line.strip().split(",")[1].split()[-1])
        r2 = float(line.strip().split()[-1])
        if ppl < min(ppls) and r2 > max(r2s):
            print("Iter %s\tPPL %.6f\tROUGE2 %.6f\t\tPPL BEST\tROUGE2 BEST")% (itr, ppl, r2)
        elif r2 > max(r2s):
            print("Iter %s\tPPL %.6f\tROUGE2 %.6f\t\t        \tROUGE2 BEST")% (itr, ppl, r2)
        elif ppl < min(ppls):
            print("Iter %s\tPPL %.6f\tROUGE2 %.6f\t\tPPL BEST")% (itr, ppl, r2)
        else:
            print("Iter %s\tPPL %.6f\tROUGE2 %.6f")% (itr, ppl, r2)
        ppls.append(ppl)
        r2s.append(r2)
    elif 'hit' in line:
        if '#' in line:
            print '######################', line.strip(), '########################'
        else:
            print '######################', line.strip()
    elif 'learning rate' in line:
        print '@@@@@@@@@@@@@ LR: ', line.strip().split()[-1]
    elif 'Start to train' in line:
        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
