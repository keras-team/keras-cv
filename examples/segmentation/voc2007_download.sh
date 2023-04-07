wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget -nc http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
mkdir voc2007_trainval
mkdir voc2007_test
tar -xvf './VOCtrainval_06-Nov-2007.tar' -C voc2007_trainval
tar -xvf './VOCtest_06-Nov-2007.tar' -C voc2007_test
